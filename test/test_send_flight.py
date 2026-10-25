from __future__ import annotations

import io
from unittest import mock

import pytest

import urllib3
from urllib3 import BodyKind, FlightPhase, RequestFlight
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import (
    HTTPConnectionPool,
    HTTPSConnectionPool,
    _one_shot_body_already_streamed,
)
from urllib3.exceptions import (
    MaxRetryError,
    UnrewindableBodyError,
)
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
from urllib3.util.request import body_to_chunks, classify_body_kind
from urllib3.util.url import Url


class TestRequestFlight:
    def test_public_namespace(self) -> None:
        assert urllib3.RequestFlight is RequestFlight
        assert urllib3.FlightPhase is FlightPhase
        assert urllib3.BodyKind is BodyKind
        assert "RequestFlight" in urllib3.__all__

    def test_phase_and_kind_values(self) -> None:
        assert [p.value for p in FlightPhase] == [
            "headers_pending",
            "headers_sent",
            "body_in_flight",
            "completed",
            "aborted",
        ]
        assert [k.value for k in BodyKind] == [
            "none",
            "bytes_like",
            "rewindable_file",
            "one_shot_iterable",
        ]

    def test_happy_path_transitions(self) -> None:
        flight = RequestFlight(
            BodyKind.BYTES_LIKE, method="POST", request_url="/"
        )
        assert flight.phase is FlightPhase.HEADERS_PENDING
        assert flight.body_kind is BodyKind.BYTES_LIKE
        assert flight.body_started is False
        assert flight.is_dirty is False
        assert flight.replayable is True
        assert flight.error is None
        assert flight.method == "POST"
        assert "headers_pending" in repr(flight)

        flight._headers_sent()
        assert flight.phase is FlightPhase.HEADERS_SENT

        flight._body_in_flight()
        assert flight.phase is FlightPhase.BODY_IN_FLIGHT
        assert flight.body_started is True
        assert flight.is_dirty is False

        flight._complete()
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_started is True
        assert flight.is_dirty is False

    def test_abort_after_body_is_dirty(self) -> None:
        flight = RequestFlight(BodyKind.ONE_SHOT_ITERABLE)
        flight._headers_sent()
        flight._body_in_flight()
        err = BrokenPipeError()
        flight._abort(err)

        assert flight.phase is FlightPhase.ABORTED
        assert flight.body_started is True
        assert flight.is_dirty is True
        assert flight.error is err
        assert flight.replayable is False

    def test_abort_in_headers_phase_is_not_body_dirty(self) -> None:
        # endheaders() failing (or a failure before the first non-empty
        # body chunk) must not be reported as a half-sent body.
        flight = RequestFlight(BodyKind.BYTES_LIKE)
        flight._abort(OSError())
        assert flight.phase is FlightPhase.ABORTED
        assert flight.body_started is False
        assert flight.is_dirty is False

        flight = RequestFlight(BodyKind.ONE_SHOT_ITERABLE)
        flight._headers_sent()
        flight._abort(OSError())
        assert flight.body_started is False
        assert flight.is_dirty is False

    def test_terminal_phases_dont_move_backwards(self) -> None:
        flight = RequestFlight(BodyKind.NONE)
        flight._complete()
        flight._abort(RuntimeError())
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.error is None

        aborted = RequestFlight(BodyKind.NONE)
        aborted._abort(RuntimeError())
        aborted._headers_sent()
        aborted._complete()
        assert aborted.phase is FlightPhase.ABORTED

    def test_read_only_public_surface(self) -> None:
        flight = RequestFlight(BodyKind.NONE)
        with pytest.raises(AttributeError):
            flight.phase = FlightPhase.COMPLETED  # type: ignore[misc]
        with pytest.raises(AttributeError):
            flight.body_kind = BodyKind.BYTES_LIKE  # type: ignore[misc]

    def test_replayable_kinds(self) -> None:
        assert RequestFlight(BodyKind.NONE).replayable is True
        assert RequestFlight(BodyKind.BYTES_LIKE).replayable is True
        assert RequestFlight(BodyKind.REWINDABLE_FILE).replayable is True
        assert RequestFlight(BodyKind.ONE_SHOT_ITERABLE).replayable is False


class TestBodyClassification:
    @pytest.mark.parametrize(
        "body,expected",
        [
            (None, BodyKind.NONE),
            (b"abc", BodyKind.BYTES_LIKE),
            ("abc", BodyKind.BYTES_LIKE),
            (bytearray(b"abc"), BodyKind.BYTES_LIKE),
            (memoryview(b"abc"), BodyKind.BYTES_LIKE),
            (io.BytesIO(b"abc"), BodyKind.REWINDABLE_FILE),
            (io.StringIO("abc"), BodyKind.REWINDABLE_FILE),
            (open(__file__, "rb"), BodyKind.REWINDABLE_FILE),
            ((b"a", b"b"), BodyKind.ONE_SHOT_ITERABLE),
            (iter([b"a", b"b"]), BodyKind.ONE_SHOT_ITERABLE),
            ((b"x" for _ in range(1)), BodyKind.ONE_SHOT_ITERABLE),
        ],
    )
    def test_classify_body_kind(self, body: object, expected: BodyKind) -> None:
        try:
            assert classify_body_kind(body) is expected
            assert body_to_chunks(body, "POST", 16).body_kind is expected
        finally:
            if hasattr(body, "close"):
                body.close()  # type: ignore[attr-defined]

    def test_invalid_body_still_type_errors(self) -> None:
        with pytest.raises(TypeError):
            body_to_chunks(object(), "POST", 16)


DEFAULT_FAKE_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"


class FakeSocket:
    """Minimal socket replacement: captures bytes, can fail on send."""

    def __init__(
        self,
        *,
        fail_headers: bool = False,
        fail_body: bool = False,
        response: bytes = DEFAULT_FAKE_RESPONSE,
    ) -> None:
        self.sent = bytearray()
        self.fail_headers = fail_headers
        self.fail_body = fail_body
        self.response = response
        self._headers_seen = False

    def settimeout(self, timeout: object) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        if self.fail_headers:
            raise BrokenPipeError("boom during headers")

        # Only raise on a sendall call that starts *after* the head already
        # went out: the single call that finishes sending the head counts as
        # a successful endheaders().
        already_complete = self._headers_seen
        self.sent += data
        if b"\r\n\r\n" in self.sent:
            self._headers_seen = True

        if self.fail_body and already_complete:
            raise BrokenPipeError("boom during body")

    def makefile(self, *args: object, **kwargs: object) -> io.BytesIO:
        return io.BytesIO(self.response)

    def close(self) -> None:
        pass


def _conn_with_fake_socket() -> HTTPConnection:
    conn = HTTPConnection("example.com")
    conn.sock = FakeSocket()
    return conn


class TestConnectionRequestFlight:
    def test_no_body_get_completes_without_body_in_flight(self) -> None:
        conn = _conn_with_fake_socket()
        assert conn.flight is None

        conn.request("GET", "/")
        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_kind is BodyKind.NONE
        assert flight.body_started is False
        assert flight.is_dirty is False

    def test_bytes_body_completes_as_body_in_flight(self) -> None:
        conn = _conn_with_fake_socket()
        conn.request("POST", "/", body=b"hello")

        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_kind is BodyKind.BYTES_LIKE
        assert flight.body_started is True
        assert bytes(conn.sock.sent).endswith(b"hello")  # type: ignore[attr-defined]

    def test_chunked_generator_completes_with_terminator(self) -> None:
        conn = _conn_with_fake_socket()
        conn.request("POST", "/", body=iter([b"ab", b"cd"]), chunked=True)

        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_kind is BodyKind.ONE_SHOT_ITERABLE
        assert flight.body_started is True
        sent = bytes(conn.sock.sent)  # type: ignore[attr-defined]
        assert b"2\r\nab\r\n" in sent
        assert b"2\r\ncd\r\n" in sent
        assert sent.endswith(b"0\r\n\r\n")

    def test_empty_chunked_body_never_enters_body_in_flight(self) -> None:
        conn = _conn_with_fake_socket()
        conn.request("POST", "/", body=iter([]), chunked=True)

        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_kind is BodyKind.ONE_SHOT_ITERABLE
        # Only the terminating chunk was sent; no non-empty body chunk.
        assert flight.body_started is False
        assert flight.is_dirty is False
        assert bytes(conn.sock.sent).endswith(b"0\r\n\r\n")  # type: ignore[attr-defined]

    def test_file_like_body_is_rewindable_kind(self) -> None:
        conn = _conn_with_fake_socket()
        conn.request("POST", "/", body=io.BytesIO(b"filebody"))

        flight = conn.flight
        assert flight is not None
        assert flight.body_kind is BodyKind.REWINDABLE_FILE
        assert flight.phase is FlightPhase.COMPLETED
        assert flight.body_started is True

    def test_broken_pipe_during_body_marks_dirty_abort(self) -> None:
        conn = HTTPConnection("example.com")
        conn.sock = FakeSocket(fail_body=True)

        with pytest.raises(BrokenPipeError):
            conn.request("POST", "/", body=b"x" * 1024)

        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.ABORTED
        assert flight.body_started is True
        assert flight.is_dirty is True
        assert isinstance(flight.error, BrokenPipeError)

    def test_failure_during_endheaders_is_aborted_but_not_dirty(self) -> None:
        conn = HTTPConnection("example.com")
        conn.sock = FakeSocket(fail_headers=True)

        with pytest.raises(BrokenPipeError):
            conn.request("POST", "/", body=b"x" * 16)

        flight = conn.flight
        assert flight is not None
        assert flight.phase is FlightPhase.ABORTED
        assert flight.body_started is False
        assert flight.is_dirty is False

    def test_new_request_replaces_flight_and_close_clears_it(self) -> None:
        conn = _conn_with_fake_socket()
        conn.request("GET", "/")
        conn.getresponse()
        first = conn.flight
        conn.request("GET", "/")
        conn.getresponse()
        second = conn.flight
        assert first is not second
        assert second is not None
        assert second.phase is FlightPhase.COMPLETED

        conn.close()
        assert conn.flight is None

    def test_connect_tunnel_does_not_create_flight(self) -> None:
        # CONNECT tunnel establishment is not part of any origin-request
        # flight: neither a successful tunnel nor a failed one may leave a
        # body-in-flight marker behind.
        conn = HTTPConnection("proxy.example", 8080)
        conn.set_tunnel("example.com", 443)
        conn.sock = FakeSocket(
            response=b"HTTP/1.1 200 Connection established\r\n\r\n"
        )
        conn._tunnel()  # type: ignore[attr-defined]

        assert conn.flight is None
        assert bytes(conn.sock.sent).startswith(b"CONNECT example.com:443")  # type: ignore[attr-defined]


class TestPutConnDirtyDiscard:
    def _pool(self) -> HTTPConnectionPool:
        return HTTPConnectionPool("example.com", maxsize=1)

    def test_dirty_connection_is_closed_and_none_refills_slot(self) -> None:
        pool = self._pool()
        pool.pool.get_nowait()  # drop the prefilled None slot

        conn = pool._new_conn()
        flight = RequestFlight(BodyKind.BYTES_LIKE)
        flight._body_in_flight()
        flight._abort(BrokenPipeError())
        conn._flight = flight

        pool._put_conn(conn)

        assert pool.pool.qsize() == 1
        assert pool.pool.get_nowait() is None
        assert conn.flight is None  # close() reset the flight

    def test_completed_connection_is_returned_as_is(self) -> None:
        pool = self._pool()
        pool.pool.get_nowait()

        conn = pool._new_conn()
        flight = RequestFlight(BodyKind.BYTES_LIKE)
        flight._headers_sent()
        flight._body_in_flight()
        flight._complete()
        conn._flight = flight

        pool._put_conn(conn)

        assert pool.pool.get_nowait() is conn

    def test_helper_truthiness(self) -> None:
        conn = HTTPConnection("example.com")
        streamed = RequestFlight(BodyKind.ONE_SHOT_ITERABLE)
        streamed._headers_sent()
        streamed._body_in_flight()
        conn._flight = streamed

        assert _one_shot_body_already_streamed(conn, iter([b"a"])) is True
        assert _one_shot_body_already_streamed(conn, None) is False

        not_started = RequestFlight(BodyKind.ONE_SHOT_ITERABLE)
        not_started._headers_sent()
        conn._flight = not_started
        assert _one_shot_body_already_streamed(conn, iter([b"a"])) is False

        bytes_flight = RequestFlight(BodyKind.BYTES_LIKE)
        bytes_flight._body_in_flight()
        conn._flight = bytes_flight
        assert _one_shot_body_already_streamed(conn, b"a") is False

        assert _one_shot_body_already_streamed(None, iter([b"a"])) is False


def _make_request_marks_flight(
    pool: HTTPConnectionPool, body_kind: BodyKind
) -> object:
    """Patch _make_request so the attempt aborts after the body started."""

    def fail(conn: HTTPConnection, *args: object, **kwargs: object) -> None:
        flight = RequestFlight(body_kind)
        flight._headers_sent()
        flight._body_in_flight()
        flight._abort(ConnectionResetError("reset"))
        conn._flight = flight
        raise OSError("connection reset")

    return mock.patch.object(pool, "_make_request", side_effect=fail)


# POST isn't in Retry's default allowed methods, so connection-error retries
# for POST have to be enabled explicitly (the exact configuration under which
# silently replaying a spent body was possible).
_RETRY_POST = Retry(
    total=3, connect=3, read=3, allowed_methods=frozenset(["POST"])
)
_RETRY_POST_ONCE = Retry(
    total=1, connect=1, read=1, allowed_methods=frozenset(["POST"])
)


class TestUrlopenRetryGuard:
    def test_one_shot_body_abort_fails_explicitly_instead_of_empty_retry(
        self,
    ) -> None:
        pool = HTTPConnectionPool("example.com")
        gen = iter([b"chunk1", b"chunk2"])

        with _make_request_marks_flight(pool, BodyKind.ONE_SHOT_ITERABLE):
            with pytest.raises(UnrewindableBodyError):
                pool.urlopen(
                    "POST",
                    "/",
                    body=gen,
                    chunked=True,
                    retries=_RETRY_POST,
                )

        # The guard fires on the recorded flight state before a second
        # attempt; with _make_request patched the body object is handed
        # through untouched, so the explicit failure (rather than a silent
        # empty replay) is the assertion above.

    def test_bytes_body_still_retries_after_abort(self) -> None:
        pool = HTTPConnectionPool("example.com")

        with _make_request_marks_flight(pool, BodyKind.BYTES_LIKE):
            with pytest.raises(MaxRetryError):
                pool.urlopen(
                    "POST",
                    "/",
                    body=b"bytesbody",
                    retries=_RETRY_POST_ONCE,
                )

    def test_file_like_body_is_not_caught_by_iterable_rule(self) -> None:
        pool = HTTPConnectionPool("example.com")
        body = io.BytesIO(b"filebody")

        # A file-like body is allowed through to the retry recursion; the
        # retry itself fails again and surfaces MaxRetryError, never
        # UnrewindableBodyError.
        with _make_request_marks_flight(pool, BodyKind.REWINDABLE_FILE):
            with pytest.raises(MaxRetryError):
                pool.urlopen(
                    "POST",
                    "/",
                    body=body,
                    retries=_RETRY_POST_ONCE,
                )

    def test_failure_before_body_starts_still_retries_generator(self) -> None:
        # A connect/headers-phase failure never touched the generator, so
        # the one-shot rule must not block the retry.
        pool = HTTPConnectionPool("example.com")
        gen = iter([b"chunk1", b"chunk2"])

        def fail_before_send(
            conn: HTTPConnection, *args: object, **kwargs: object
        ) -> None:
            # No flight attached: conn.request() was never reached.
            raise OSError("connect failed")

        with mock.patch.object(pool, "_make_request", side_effect=fail_before_send):
            with pytest.raises(MaxRetryError):
                pool.urlopen(
                    "POST",
                    "/",
                    body=gen,
                    chunked=True,
                    retries=_RETRY_POST_ONCE,
                )

        # The generator was never iterated and still holds all its data.
        assert next(gen) == b"chunk1"

    def test_connect_tunnel_failure_does_not_taint_body(self) -> None:
        # HTTPS through an HTTP proxy: the CONNECT handshake fails before any
        # origin-request flight exists. A one-shot body must stay untouched
        # and the failure must not be reported as a half-sent body retry.
        pool = HTTPSConnectionPool(
            "example.com",
            _proxy=Url(scheme="http", host="127.0.0.1", port=1),
        )
        gen = iter([b"chunk1"])

        with mock.patch.object(pool, "_prepare_proxy", side_effect=OSError("nope")):
            with pytest.raises(MaxRetryError) as exc_info:
                pool.urlopen(
                    "POST",
                    "/",
                    body=gen,
                    chunked=True,
                    retries=Retry(
                        0, allowed_methods=frozenset(["POST"])
                    ),
                )

        assert not isinstance(exc_info.value, UnrewindableBodyError)
        # Body never made it anywhere.
        assert next(gen) == b"chunk1"


class TestPoolManagerCrossHostRedirectGuard:
    def test_one_shot_body_cross_host_redirect_fails_explicitly(self) -> None:
        # The pool layer cannot see the flight across pools, so PoolManager
        # must consult the flight recorded on the response and refuse before
        # following a cross-host redirect with a spent one-shot body.
        flight = RequestFlight(BodyKind.ONE_SHOT_ITERABLE)
        flight._headers_sent()
        flight._body_in_flight()
        flight._complete()

        class FakeResponse:
            status = 307
            retries = Retry(total=10, allowed_methods=frozenset(["POST"]))
            _request_flight = flight

            def get_redirect_location(self) -> str:
                return "http://other.example/"

            def drain_conn(self) -> None:
                pass

        class FakePool:
            scheme = "http"
            host = "first.example"
            port = 80

            def urlopen(self, *args: object, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def is_same_host(self, url: object) -> bool:
                return False

        manager = PoolManager()
        fake_pool = FakePool()
        calls = 0

        def connection_from_host(*args: object, **kwargs: object) -> FakePool:
            nonlocal calls
            calls += 1
            return fake_pool

        with mock.patch.object(
            manager, "connection_from_host", side_effect=connection_from_host
        ):
            with pytest.raises(UnrewindableBodyError):
                manager.urlopen(
                    "POST",
                    "http://first.example/",
                    body=iter([b"chunk1"]),
                    chunked=True,
                    retries=Retry(
                        total=10, allowed_methods=frozenset(["POST"])
                    ),
                )

        # The redirected request to the second host was never started.
        assert calls == 1

    def test_file_like_body_cross_host_redirect_is_not_blocked(self) -> None:
        # File-like bodies replay through body_pos across pools as before.
        first_flight = RequestFlight(BodyKind.REWINDABLE_FILE)
        first_flight._headers_sent()
        first_flight._body_in_flight()
        first_flight._complete()

        second_flight = RequestFlight(BodyKind.REWINDABLE_FILE)
        second_flight._headers_sent()
        second_flight._body_in_flight()
        second_flight._complete()

        class FirstResponse:
            status = 307
            retries = Retry(total=10, allowed_methods=frozenset(["POST"]))
            _request_flight = first_flight

            def get_redirect_location(self) -> str:
                return "http://other.example/ok"

            def drain_conn(self) -> None:
                pass

        class FinalResponse:
            status = 200
            retries = Retry(total=10, allowed_methods=frozenset(["POST"]))
            _request_flight = second_flight

            def get_redirect_location(self) -> None:
                return None

            def drain_conn(self) -> None:
                pass

        class FakePool:
            def __init__(self, response: object, host: str) -> None:
                self._response = response
                self.scheme = "http"
                self.host = host
                self.port = 80

            def urlopen(self, *args: object, **kwargs: object) -> object:
                return self._response

            def is_same_host(self, url: object) -> bool:
                return False

        manager = PoolManager()
        pools = [
            FakePool(FirstResponse(), "first.example"),
            FakePool(FinalResponse(), "other.example"),
        ]

        with mock.patch.object(
            manager, "connection_from_host", side_effect=pools
        ):
            response = manager.urlopen(
                "POST",
                "http://first.example/",
                body=io.BytesIO(b"filebody"),
                chunked=True,
                retries=Retry(
                    total=10, allowed_methods=frozenset(["POST"])
                ),
            )

        assert response.status == 200
