import httpx
from langchain_core.tools import tool


@tool
def http_request(
    method: str,
    url: str,
    headers: dict = None,
    body: str = None,
) -> str:
    """Make an HTTP request to any URL on the internet.

    Use this to fetch web pages, call APIs, or retrieve documents from a URL.

    Args:
        method: HTTP method — GET, POST, PUT, DELETE, PATCH, HEAD, or OPTIONS
        url: Full URL including scheme (e.g. https://example.com/api)
        headers: Optional HTTP headers as key-value pairs
        body: Optional request body as a plain string

    Returns:
        HTTP status code followed by the response body (capped at 5000 characters)
    """
    try:
        response = httpx.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            content=body.encode() if body else None,
            timeout=30,
            follow_redirects=True,
        )
        return f"Status: {response.status_code}\n\n{response.text[:5000]}"
    except Exception as e:
        return f"Error making {method.upper()} request to {url}: {e}"
