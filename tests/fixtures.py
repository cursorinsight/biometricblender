from pytest import fixture
from tempfile import NamedTemporaryFile


@fixture
def temppath() -> str:
    with NamedTemporaryFile() as file:
        yield file.name
