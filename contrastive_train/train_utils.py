import torch

def save_model_hook(models, weights, output_dir):
    for i, model in enumerate(models):
        model.save_pretrained(output_dir, state_dict=weights[i])
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
            model.load_adapter(input_dir, model.active_adapter, is_trainable=True)
            
# def get_cosine_sim(query_embs, passage_embs, logit_scale):
#     return torch.matmul(query_embs, passage_embs.t()) *logit_scale


# def get_nt_xent_loss(sim_scores):
#     # Took from: https://github.com/UKPLab/sentence-transformers/blob/master
#     # /sentence_transformers/losses/MultipleNegativesRankingLoss.py 
#     """
#     Compute the cross_entropy given the square similarity matrix.
#     This indicates that `a_i` and `b_j` have high similarity
#     when `i==j` and low similarity when `i!=j`.
#     """
#     return torch.nn.functional.cross_entropy(sim_scores, torch.arange(len(sim_scores), device=sim_scores.device))


def get_cosing_embeddings(query_embs, product_embs):
    return torch.sum(query_embs * product_embs, axis=1)


def get_loss(cosine_score, labels):
    """
    Computes a custom contrastive loss based on cosine similarity and binary labels.

    Args:
        cosine_score (torch.Tensor): A tensor representing the cosine similarity scores between embeddings.
        labels (torch.Tensor): A tensor of binary labels (0 or 1) indicating matching (1) or non-matching (0) cases.

    Returns:
        torch.Tensor: The computed contrastive loss value, promoting similarity for matching cases and dissimilarity for non-matching cases.
    """
    return torch.mean(torch.square(labels * (1 - cosine_score) + torch.clamp((1 - labels) * cosine_score, min=0.0)))