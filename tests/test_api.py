"""API endpoint tests"""

from api import server as server_module


def test_server_alias_is_available_for_asgi_imports():
    """Railway-style imports should find a server object on the module."""
    assert server_module.app is server_module.server
    assert callable(server_module.app)
