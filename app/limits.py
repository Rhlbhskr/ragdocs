"""Reject oversized bodies before multipart parsing consumes disk space."""
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'detail': 'Invalid content length.'}, 400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({'detail': 'Request exceeds upload size limit.'}, 413)(scope, receive, send)
        # Bounded buffering also covers chunked requests without Content-Length.
        messages, total = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            total += len(message.get('body', b''))
            if total > self.max_bytes:
                return await JSONResponse({'detail': 'Request exceeds upload size limit.'}, 413)(scope, receive, send)
            messages.append(message)
            if not message.get('more_body', False):
                break
        async def buffered_receive():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, buffered_receive, send)
