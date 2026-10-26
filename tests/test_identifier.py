import pytest
from freshpick.identifier import calibrate_margin


def test_margin_calibration_balances_domains_instead_of_sample_counts():
    result=calibrate_margin({'web':[.1,.2],'dataset':[.01,.02,.03]},[-.2,-.1])
    threshold=result['margin_threshold']
    assert -.1 < threshold < .01
    assert result['known_domain_acceptance']=={'web':1.0,'dataset':1.0}
    assert result['ood_rejection']==1.0
    # Small numerical changes around observed scores cannot flip a midpoint decision.
    assert threshold != .01 and threshold != -.1


def test_margin_calibration_rejects_empty_validation_groups():
    with pytest.raises(ValueError):calibrate_margin({'web':[]},[-.1])
