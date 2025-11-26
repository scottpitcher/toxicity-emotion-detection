import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb


# ======================================================
#  OLD BASELINE MODEL (matches your saved checkpoint)
# ======================================================
class OldBaselineBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_toxicity_labels=6):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_toxicity_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


# ======================================================
#  INLINE SENTENCE EXPLAINER FOR OLD MODEL
# ======================================================
class OldModelExplainer:
    """Inline token-level explainer for OldBaselineBERT."""

    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased"):
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.model.to(device)

        # Fixed toxicity class order
        self.toxicity_types = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate'
        ]

        # Fixed colors per class (base hues)
        self.class_colors = {
            'toxic'        : '#e41a1c',  # red
            'severe_toxic' : '#a50f15',  # darker red
            'obscene'      : '#984ea3',  # purple
            'threat'       : '#377eb8',  # blue
            'insult'       : '#ff7f00',  # orange
            'identity_hate': '#4daf4a',  # green
        }

    # ----------------------------------------------
    # Class-wise gradient attributions
    # ----------------------------------------------
    def get_token_attributions(self, text: str):
        """
        Returns:
            tokens_clean: list[str] (no CLS/SEP/PAD)
            saliency_per_token_class: np.array [num_tokens, num_classes]
            logits: torch.Tensor [num_classes]
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Forward once to get logits & to know sequence length
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_output = outputs.last_hidden_state[:, 0, :]
            logits = self.model.classifier(cls_output)[0]  # (6,)

        # Now recompute with grad-enabled embeddings for attribution
        embedding_layer = self.model.bert.embeddings.word_embeddings
        embeddings = embedding_layer(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        outputs_grad = self.model.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        cls_output_grad = outputs_grad.last_hidden_state[:, 0, :]
        logits_grad = self.model.classifier(cls_output_grad)[0]  # (6,)

        num_classes = logits_grad.shape[-1]

        # saliency_per_class: [num_classes, seq_len]
        saliency_per_class = []

        for c in range(num_classes):
            # Clear previous grads
            if embeddings.grad is not None:
                embeddings.grad.zero_()

            logit_c = logits_grad[c]
            logit_c.backward(retain_graph=True)

            # embeddings.grad: [1, seq_len, hidden]
            sal_c = embeddings.grad.abs().mean(dim=-1).detach().cpu().numpy()[0]
            saliency_per_class.append(sal_c)

        saliency_per_class = np.stack(saliency_per_class, axis=0)  # [C, L]

        # Decode tokens, trim to actual length
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        actual_length = int(attention_mask.sum().item())

        tokens = tokens[:actual_length]
        saliency_per_class = saliency_per_class[:, :actual_length]  # [C, L]

        # Filter out special tokens and padding
        keep_idx = [
            i for i, tok in enumerate(tokens)
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        tokens_clean = [tokens[i] for i in keep_idx]
        saliency_per_class = saliency_per_class[:, keep_idx]  # [C, T]

        # Transpose to [T, C]
        saliency_per_token_class = saliency_per_class.T

        return tokens_clean, saliency_per_token_class, logits.detach().cpu()

    # ----------------------------------------------
    # Helper: mix base class color with white
    # ----------------------------------------------
    def _color_for_token(self, class_idx: int, intensity: float):
        """
        intensity in [0,1]: 0 = white, 1 = pure class color
        """
        class_name = self.toxicity_types[class_idx]
        base_rgb = np.array(to_rgb(self.class_colors[class_name]))
        white = np.array([1.0, 1.0, 1.0])
        rgb = (1 - intensity) * white + intensity * base_rgb
        return tuple(rgb)

    # ----------------------------------------------
    # Visualization
    # ----------------------------------------------
    def visualize_inline(
        self,
        text: str,
        top_k_tokens: int = 4,
        top_k_classes: int = 3,
        output_path: str = None,
    ):
        """
        Shows sentence inline with tokens colored by most-influenced class.
        Under each influential token, prints its top-k classes & scores.
        """
        tokens, sal_tok_class, logits = self.get_token_attributions(text)
        num_tokens, num_classes = sal_tok_class.shape

        # Global prediction probs
        with torch.no_grad():
            probs = torch.sigmoid(logits).numpy()

        # For each token: max saliency over classes (for intensity)
        max_saliency_per_token = sal_tok_class.max(axis=1)  # [T]
        # Normalize intensities to [0,1]
        if max_saliency_per_token.max() > 0:
            intensities = max_saliency_per_token / (max_saliency_per_token.max() + 1e-8)
        else:
            intensities = np.zeros_like(max_saliency_per_token)

        # For each token: class index with max saliency
        best_class_idx = sal_tok_class.argmax(axis=1)  # [T]

        # Choose top-k most influential tokens overall
        top_token_indices = np.argsort(max_saliency_per_token)[-top_k_tokens:][::-1]

        fig, ax = plt.subplots(figsize=(14, 4))

        # Draw inline sentence as boxes
        x_pos = 0
        for i, tok in enumerate(tokens):
            intensity = float(intensities[i])
            c_idx = int(best_class_idx[i])
            face_rgb = self._color_for_token(c_idx, intensity)

            rect = patches.Rectangle(
                (x_pos, 0.0), 1.0, 1.0,
                linewidth=1,
                edgecolor='black',
                facecolor=face_rgb
            )
            ax.add_patch(rect)

            ax.text(
                x_pos + 0.5, 0.5, tok,
                ha='center', va='center',
                fontsize=11, fontweight='bold'
            )

            # If this token is among the top influential tokens, annotate under it
            if i in top_token_indices:
                # top-k classes for this token
                sal_this = sal_tok_class[i]  # [C]
                top_classes = np.argsort(sal_this)[-top_k_classes:][::-1]
                lines = []
                for c in top_classes:
                    cname = self.toxicity_types[c]
                    score = sal_this[c]
                    lines.append(f"{cname}: {score:.3f}")
                text_block = "\n".join(lines)

                ax.text(
                    x_pos + 0.5, -0.25,
                    text_block,
                    ha='center', va='top',
                    fontsize=8
                )

            x_pos += 1

        ax.set_xlim(0, x_pos)
        ax.set_ylim(-0.8, 1.1)
        ax.axis('off')
        ax.set_title("Token-Level Attributions (color = most influenced class, intensity = strength)")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

        return fig

    # ----------------------------------------------
    # Console summary
    # ----------------------------------------------
    def print_explanation(self, text: str, top_k_tokens: int = 5):
        tokens, sal_tok_class, logits = self.get_token_attributions(text)
        probs = torch.sigmoid(logits).numpy()

        print("\n=== TOXICITY EXPLANATION ===")
        print(f"Input: {text}\n")

        print("Overall prediction probs:")
        for i, cname in enumerate(self.toxicity_types):
            print(f"{cname:15} {probs[i]:.4f}")

        max_saliency_per_token = sal_tok_class.max(axis=1)
        top_token_indices = np.argsort(max_saliency_per_token)[-top_k_tokens:][::-1]

        print("\nMost influential tokens:")
        for idx in top_token_indices:
            tok = tokens[idx]
            sal_row = sal_tok_class[idx]
            top_classes = np.argsort(sal_row)[-3:][::-1]
            cls_str = ", ".join(
                f"{self.toxicity_types[c]}: {sal_row[c]:.4f}" for c in top_classes
            )
            print(f"  {tok:10} -> {cls_str}")

        print("================================\n")


# ======================================================
#  MAIN
# ======================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load old checkpoint
    model = OldBaselineBERT()
    checkpoint = torch.load("../models/baseline_final.pt", map_location=device)
    model.load_state_dict(checkpoint)

    explainer = OldModelExplainer(model, device=device)

    test_text = "I am gonna hurt you"
    explainer.print_explanation(test_text)
    explainer.visualize_inline(test_text, top_k_tokens=4, top_k_classes=3)
