"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

# config/wsgi.py
import os
from django.core.wsgi import get_wsgi_application
from dotenv import load_dotenv

load_dotenv()

application = get_wsgi_application()
