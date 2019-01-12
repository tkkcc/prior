from importlib import import_module
from config import o

Model = import_module("." + o.model).ModelStack
