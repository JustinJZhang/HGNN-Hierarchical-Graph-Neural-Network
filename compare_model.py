import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DROP_OUT_RATE, Predictor, SCALE


PATH_EMBEDDING_SIZE = int(256 * SCALE)
BACKBONE_OUTPUT_SIZE = 2 * PATH_EMBEDDING_SIZE


class ComparisonModel(nn.Module):
    """Provide the shared dual-task prediction interface for DL baselines.

    Args:
        num_AU: Integer number of action-unit features per expression.
        num_of_fa_feat: Integer number of asymmetry features per expression.
        num_facial_expressions: Integer number of expressions per subject.
        device: PyTorch device used by the experiment.

    Input:
        Each subclass accepts a floating-point tensor with shape
        [batch_size * num_facial_expressions, num_AU + num_of_fa_feat].

    Output:
        Floating-point logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
    ):
        super().__init__()
        self.num_features = num_AU + num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.device = device
        self.severity_head = Predictor()
        self.asymmetry_head = Predictor()

    def _predict(self, features):
        """Map shared subject features [B, 256] to logits [B, 2, 3]."""
        feature_1, feature_2 = torch.split(
            features,
            [PATH_EMBEDDING_SIZE, PATH_EMBEDDING_SIZE],
            dim=1,
        )
        severity = self.severity_head(feature_1, feature_2)
        asymmetry = self.asymmetry_head(feature_1, feature_2)
        return torch.stack((severity, asymmetry), dim=1)

class CNN_Model(ComparisonModel):
    """Encode each subject with a three-layer 2D CNN.

    Args:
        num_AU: Integer number of AU features; expected value 17.
        num_of_fa_feat: Integer number of FA features; expected value 24.
        num_facial_expressions: Integer expression count; expected value 8.
        device: PyTorch device used by the experiment.

    Input:
        Floating-point tensor with shape [batch_size * 8, 41].

    Output:
        Floating-point severity/asymmetry logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
    ):
        super().__init__(
            num_AU,
            num_of_fa_feat,
            num_facial_expressions,
            device,
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, BACKBONE_OUTPUT_SIZE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.shape[0] // self.num_facial_expressions
        x = x.reshape(
            batch_size,
            1,
            self.num_facial_expressions,
            self.num_features,
        )
        return self._predict(self.backbone(x))


class CNN1D_Model(ComparisonModel):
    """Encode the eight-expression sequence with a three-layer 1D CNN.

    Args:
        num_AU: Integer number of AU features; expected value 17.
        num_of_fa_feat: Integer number of FA features; expected value 24.
        num_facial_expressions: Integer sequence length; expected value 8.
        device: PyTorch device used by the experiment.

    Input:
        Floating-point tensor with shape [batch_size * 8, 41].

    Output:
        Floating-point severity/asymmetry logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
    ):
        super().__init__(
            num_AU,
            num_of_fa_feat,
            num_facial_expressions,
            device,
        )
        self.backbone = nn.Sequential(
            nn.Conv1d(self.num_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, BACKBONE_OUTPUT_SIZE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.shape[0] // self.num_facial_expressions
        x = x.reshape(
            batch_size,
            self.num_facial_expressions,
            self.num_features,
        )
        x = x.permute(0, 2, 1)
        return self._predict(self.backbone(x))


class MLP_Model(ComparisonModel):
    """Encode flattened subject features with a feed-forward neural network.

    Args:
        num_AU: Integer number of AU features; expected value 17.
        num_of_fa_feat: Integer number of FA features; expected value 24.
        num_facial_expressions: Integer expression count; expected value 8.
        device: PyTorch device used by the experiment.

    Input:
        Floating-point tensor with shape [batch_size * 8, 41].

    Output:
        Floating-point severity/asymmetry logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
    ):
        super().__init__(
            num_AU,
            num_of_fa_feat,
            num_facial_expressions,
            device,
        )
        input_size = num_facial_expressions * self.num_features
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(DROP_OUT_RATE),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROP_OUT_RATE),
            nn.Linear(512, BACKBONE_OUTPUT_SIZE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.shape[0] // self.num_facial_expressions
        x = x.reshape(batch_size, -1)
        return self._predict(self.backbone(x))


class NumericalFeatureTokenizer(nn.Module):
    """Tokenize numerical features across expression contexts.

    Args:
        num_features: Integer number of features measured per expression.
        num_expressions: Integer number of expression contexts per subject.
        token_size: Integer embedding size assigned to every feature.

    Input:
        Floating-point tensor with shape
        [batch_size, num_expressions, num_features].

    Output:
        Floating-point tokens with shape
        [batch_size, num_expressions * num_features, token_size].
    """

    def __init__(self, num_features, num_expressions, token_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, token_size))
        self.bias = nn.Parameter(torch.empty(num_features, token_size))
        self.expression_embedding = nn.Parameter(
            torch.empty(num_expressions, token_size)
        )
        bound = 1.0 / math.sqrt(token_size)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.expression_embedding, -bound, bound)

    def forward(self, x):
        tokens = (
            x.unsqueeze(-1) * self.weight.reshape(1, 1, *self.weight.shape)
            + self.bias.reshape(1, 1, *self.bias.shape)
            + self.expression_embedding.reshape(
                1,
                self.expression_embedding.shape[0],
                1,
                self.expression_embedding.shape[1],
            )
        )
        return tokens.flatten(1, 2)


class ReGLU(nn.Module):
    def forward(self, x):
        values, gates = x.chunk(2, dim=-1)
        return values * F.relu(gates)


class FTTransformerBlock(nn.Module):
    """Apply one FT-Transformer block.

    Args:
        token_size: Integer feature-token dimension.
        num_heads: Integer number of self-attention heads.
        ffn_hidden_size: Integer ReGLU feed-forward hidden dimension.
        attention_dropout: Floating-point attention dropout probability.
        ffn_dropout: Floating-point feed-forward dropout probability.
        residual_dropout: Floating-point residual-branch dropout probability.
        use_attention_norm: Boolean indicating whether to normalize attention input.
        cls_only: Boolean indicating whether only the final CLS query is evaluated.

    Input:
        Floating-point tokens with shape [batch_size, num_tokens, token_size].

    Output:
        Floating-point tokens with shape [batch_size, num_tokens, token_size],
        or [batch_size, 1, token_size] when cls_only is True.
    """

    def __init__(
        self,
        token_size,
        num_heads,
        ffn_hidden_size,
        attention_dropout,
        ffn_dropout,
        residual_dropout,
        use_attention_norm,
        cls_only,
    ):
        super().__init__()
        self.attention_norm = (
            nn.LayerNorm(token_size) if use_attention_norm else nn.Identity()
        )
        self.cls_only = cls_only
        self.attention = nn.MultiheadAttention(
            token_size,
            num_heads,
            dropout=attention_dropout,
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_norm = nn.LayerNorm(token_size)
        self.ffn = nn.Sequential(
            nn.Linear(token_size, 2 * ffn_hidden_size),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_size, token_size),
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x):
        attention_input = self.attention_norm(x)
        query = attention_input[:, -1:, :] if self.cls_only else attention_input
        attended, _ = self.attention(
            query.transpose(0, 1),
            attention_input.transpose(0, 1),
            attention_input.transpose(0, 1),
            need_weights=False,
        )
        residual = x[:, -1:, :] if self.cls_only else x
        x = residual + self.attention_residual_dropout(attended.transpose(0, 1))
        x = x + self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FT_Transformer_Model(ComparisonModel):
    """Encode all 8x41 numerical features with a FT-Transformer.

    Args:
        num_AU: Integer number of AU features; expected value 17.
        num_of_fa_feat: Integer number of FA features; expected value 24.
        num_facial_expressions: Integer expression count; expected value 8.
        device: PyTorch device used by the experiment.
        token_size: Integer feature-token dimension; default 192.
        num_heads: Integer attention-head count; default 8.
        num_layers: Integer transformer-block count; default 3.
        ffn_factor: Floating-point ReGLU hidden-size factor; default 4/3.
        attention_dropout: Floating-point attention dropout; default 0.2.
        ffn_dropout: Floating-point feed-forward dropout; default 0.1.
        residual_dropout: Floating-point residual dropout; default 0.0.

    Input:
        Floating-point tensor with shape [batch_size * 8, 41].

    Output:
        Floating-point severity/asymmetry logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
        token_size=192,
        num_heads=8,
        num_layers=3,
        ffn_factor=4.0 / 3.0,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__(
            num_AU,
            num_of_fa_feat,
            num_facial_expressions,
            device,
        )
        ffn_hidden_size = int(token_size * ffn_factor)
        self.feature_tokenizer = NumericalFeatureTokenizer(
            self.num_features,
            num_facial_expressions,
            token_size,
        )
        self.cls_token = nn.Parameter(torch.empty(token_size))
        bound = 1.0 / math.sqrt(token_size)
        nn.init.uniform_(self.cls_token, -bound, bound)
        self.blocks = nn.ModuleList(
            FTTransformerBlock(
                token_size,
                num_heads,
                ffn_hidden_size,
                attention_dropout,
                ffn_dropout,
                residual_dropout,
                use_attention_norm=layer_index > 0,
                cls_only=layer_index == num_layers - 1,
            )
            for layer_index in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(token_size)
        self.output_projection = nn.Linear(
            token_size,
            BACKBONE_OUTPUT_SIZE,
        )

    def forward(self, x):
        batch_size = x.shape[0] // self.num_facial_expressions
        x = x.reshape(
            batch_size,
            self.num_facial_expressions,
            self.num_features,
        )
        tokens = self.feature_tokenizer(x)
        cls_tokens = self.cls_token.reshape(1, 1, -1).expand(
            batch_size,
            1,
            -1,
        )
        tokens = torch.cat((tokens, cls_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        cls_features = F.relu(self.output_norm(tokens[:, -1, :]))
        features = self.output_projection(cls_features)
        return self._predict(features)
