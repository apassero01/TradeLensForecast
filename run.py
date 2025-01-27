import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')
django.setup()

from daphne.cli import CommandLineInterface

if __name__ == '__main__':
    cli = CommandLineInterface()
    cli.run(['-b', '0.0.0.0', '-p', '8000', 'TradeLens.asgi:application']) 