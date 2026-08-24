"""Test suite for ``aegis_ml``.

A package rather than a bare directory so ``tests.fixtures`` is importable by name from
every test module, which is what keeps the mocks-only-in-fixtures rule enforceable: there
is exactly one import path a test double can come from.
"""
