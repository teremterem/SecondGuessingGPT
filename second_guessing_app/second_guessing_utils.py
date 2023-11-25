"""
This module contains utility functions for the second_guessing_app.
"""
from functools import wraps


def csrf_exempt_async(view_func):
    """Decorator to mark a view as csrf exempt (for async views)."""

    @wraps(view_func)
    async def wrapped_view(*args, **kwargs):
        """The wrapper of the view function."""
        return await view_func(*args, **kwargs)

    wrapped_view.csrf_exempt = True
    return wrapped_view
