from genesis.utils.uid import UID


def test_match_accepts_the_displayed_uid():
    uid = UID()
    assert uid.match(uid.uid)
    assert uid.match(uid.full())
    assert uid.match(uid.short(), short_only=True)
    assert not uid.match(uid.short())
    assert not uid.match(uid.full(), short_only=True)
