from hyperas.distributions import conditional


def test_conditional():
    data = 'foo'
    assert data == conditional(data)
