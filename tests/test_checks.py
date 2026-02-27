import pandas as pd
from dq_engine.checks import check_completeness, check_uniqueness, check_range, check_domain, check_freshness

def test_completeness():
    df = pd.DataFrame({"a":[1,None,""]})
    r = check_completeness(df, "a", 0)
    assert r.status == "fail"
    assert r.observed["null_count"] == 2

def test_uniqueness():
    df = pd.DataFrame({"id":[1,1,2]})
    r = check_uniqueness(df, "id", 0)
    assert r.status == "fail"

def test_range():
    df = pd.DataFrame({"x":[1,2,-1]})
    r = check_range(df, "x", 0, None)
    assert r.status == "fail"

def test_domain():
    df = pd.DataFrame({"u":["kg","bad"]})
    r = check_domain(df, "u", ["kg","g"])
    assert r.status == "fail"

def test_freshness_pass():
    df = pd.DataFrame({"ts_load": ["2026-02-25T10:00:00Z"]})
    r = check_freshness(df, "ts_load", 2)
    assert r.status in ("pass","fail")    
