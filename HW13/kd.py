# Knowledge distillation loss (HW13.ipynb cell [24]; the TODO is filled in)
import torch.nn.functional as F


# Implement the loss function with KL divergence loss for knowledge distillation.
# You also have to copy-paste this whole block to HW13 GradeScope.
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.5, temperature=1.0):
    # ------------TODO-------------
    # Refer to the above formula and finish the loss function for knowkedge distillation using KL divergence loss and CE loss.
    # If you have no idea, please take a look at the provided useful link above.
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    # kl_div(input, target) = sum target * (log target - input), averaged over the batch
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * temperature ** 2 * kd_loss + (1 - alpha) * ce_loss
