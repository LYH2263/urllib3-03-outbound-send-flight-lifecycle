from __future__ import annotations

import enum


class FlightPhase(enum.Enum):
    """
    Lifecycle phase of one outbound request send
    (:class:`RequestFlight`).

    Transitions are strictly forward, apart from ``ABORTED`` which can be
    entered from any other phase:

    ``HEADERS_PENDING`` -> ``HEADERS_SENT`` -> ``BODY_IN_FLIGHT``
    -> ``COMPLETED``

    A failure while sending jumps to ``ABORTED``.
    """

    #: The flight was created but the request head hasn't fully left
    #: the client yet (``endheaders()`` hasn't successfully returned).
    HEADERS_PENDING = "headers_pending"

    #: The request line and headers were successfully sent, no body byte
    #: is in flight yet.
    HEADERS_SENT = "headers_sent"

    #: At least one non-empty body chunk started being sent.
    BODY_IN_FLIGHT = "body_in_flight"

    #: The whole request was sent, including the chunked terminating
    #: empty chunk when applicable.
    COMPLETED = "completed"

    #: Sending the request failed. Whether any body byte made it to the
    #: wire is available through :attr:`RequestFlight.body_started`.
    ABORTED = "aborted"


class BodyKind(enum.Enum):
    """
    How the request body of a :class:`RequestFlight` is supplied, which
    determines whether it can be sent again on a retry or redirect.
    """

    #: ``body is None``.
    NONE = "none"

    #: ``bytes``/``str`` or any buffer-protocol object. Safe to re-send.
    BYTES_LIKE = "bytes_like"

    #: A file-like object (``read()``). Re-send goes through the existing
    #: ``body_pos``/``seek()`` rewind machinery.
    REWINDABLE_FILE = "rewindable_file"

    #: A one-shot iterable/generator. Once the body started being sent the
    #: iterable is consumed and must not be silently iterated again.
    ONE_SHOT_ITERABLE = "one_shot_iterable"


#: Phases that are terminal; no further forward transition is allowed.
_TERMINAL_PHASES = frozenset((FlightPhase.COMPLETED, FlightPhase.ABORTED))

#: Body kinds the pool is allowed to send again on retry/redirect.
_REPLAYABLE_BODY_KINDS = frozenset(
    (
        BodyKind.NONE,
        BodyKind.BYTES_LIKE,
        BodyKind.REWINDABLE_FILE,
    )
)


class RequestFlight:
    """
    Outbound request-send flight.

    A flight tracks the lifecycle of **one**
    :meth:`urllib3.connection.HTTPConnection.request` send on the
    connection it is attached to. It is the first-class counterpart of the
    temporary, response-side option bag: while the response options only
    describe how the response is meant to be read, the flight describes how
    far the outbound request actually got onto the wire.

    The object is read-only for callers and middleware. The connection send
    loop drives the internal transitions:

    - ``endheaders()`` returning successfully moves the flight from
      :attr:`FlightPhase.HEADERS_PENDING` to
      :attr:`FlightPhase.HEADERS_SENT`.
    - Sending the first non-empty body chunk moves it to
      :attr:`FlightPhase.BODY_IN_FLIGHT`.
    - Finishing the send normally (including the chunked terminating chunk)
      moves it to :attr:`FlightPhase.COMPLETED`.
    - Any exception raised while sending moves it to
      :attr:`FlightPhase.ABORTED`.

    The CONNECT tunnel handshake is deliberately *not* part of a flight:
    no flight exists while the tunnel is established, and a tunnel failure
    therefore never looks like a half-sent request body. The origin request
    after a successful tunnel starts its own flight.
    """

    __slots__ = (
        "_body_kind",
        "_body_started",
        "_error",
        "_phase",
        "method",
        "request_url",
    )

    def __init__(
        self,
        body_kind: BodyKind,
        *,
        method: str | None = None,
        request_url: str | None = None,
    ) -> None:
        self._phase = FlightPhase.HEADERS_PENDING
        self._body_kind = body_kind
        self._body_started = False
        self._error: BaseException | None = None
        self.method = method
        self.request_url = request_url

    # -- Public, read-only surface -------------------------------------

    @property
    def phase(self) -> FlightPhase:
        """Current :class:`FlightPhase` of the send."""
        return self._phase

    @property
    def body_kind(self) -> BodyKind:
        """The :class:`BodyKind` of the request being sent."""
        return self._body_kind

    @property
    def body_started(self) -> bool:
        """
        Whether sending a non-empty body chunk was ever started.

        Stays ``True`` for ``BODY_IN_FLIGHT``, ``COMPLETED`` and ``ABORTED``
        reached after the body started, so it survives the phase moving on.
        """
        return self._body_started

    @property
    def is_dirty(self) -> bool:
        """
        Whether the send aborted after a body chunk started being written.

        A dirty connection may still carry a salvageable response on the
        wire, but its request/response framing can't be trusted for another
        exchange, so it must be closed and discarded instead of being
        returned to the pool as a keep-alive connection.
        """
        return self._phase is FlightPhase.ABORTED and self._body_started

    @property
    def replayable(self) -> bool:
        """
        Whether the body can be sent again on a retry or redirect without
        rewinding (``none`` and ``bytes_like``) or through the existing
        file-position rewind machinery (``rewindable_file``).
        """
        return self._body_kind in _REPLAYABLE_BODY_KINDS

    @property
    def error(self) -> BaseException | None:
        """The exception that aborted the send, if any."""
        return self._error

    # -- Internal transition surface -----------------------------------
    # Driven by the connection send loop. Not part of the read-only public
    # contract; named with a leading underscore to make that explicit.

    def _advance(self, phase: FlightPhase) -> None:
        """Move to ``phase`` unless the flight is already terminal."""
        if self._phase in _TERMINAL_PHASES:
            return
        self._phase = phase

    def _headers_sent(self) -> None:
        self._advance(FlightPhase.HEADERS_SENT)

    def _body_in_flight(self) -> None:
        self._body_started = True
        self._advance(FlightPhase.BODY_IN_FLIGHT)

    def _complete(self) -> None:
        self._advance(FlightPhase.COMPLETED)

    def _abort(self, error: BaseException | None = None) -> None:
        # A completed send is terminal: nothing that happens afterwards
        # (e.g. exceptions while unwinding) retroactively aborts it.
        if self._phase in (FlightPhase.ABORTED, FlightPhase.COMPLETED):
            return
        if error is not None:
            self._error = error
        self._phase = FlightPhase.ABORTED

    def __repr__(self) -> str:
        return (
            f"<RequestFlight phase={self._phase.value} "
            f"body_kind={self._body_kind.value}"
            f" body_started={self._body_started}>"
        )
