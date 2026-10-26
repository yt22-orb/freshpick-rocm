import torch

from freshpick.train import HierarchicalLoss, classification_metrics, hierarchy_indices


NAMES = [f"{stage}_{produce}" for stage in ("fresh", "semi_fresh", "rotten")
         for produce in ("banana", "tomato")]


def test_hierarchical_loss_is_finite_and_backpropagates():
    logits = torch.randn(4, len(NAMES), requires_grad=True)
    targets = torch.tensor([0, 1, 2, 5])
    loss = HierarchicalLoss(NAMES)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_metrics_use_marginal_predictions_for_hierarchy():
    hierarchy = hierarchy_indices(NAMES)
    metrics = classification_metrics(
        labels=[0, 1], predictions=[0, 1], hierarchy=hierarchy,
        produce_predictions=[0, 1], freshness_predictions=[0, 0])
    assert metrics["accuracy"] == 1
    assert metrics["produce_accuracy"] == 1
    assert metrics["freshness_accuracy"] == 1
    assert 0 <= metrics["selection_score"] <= 1
