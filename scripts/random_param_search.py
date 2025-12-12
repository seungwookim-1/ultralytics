from dataclasses import dataclass
import random, math
from typing import Dict


def uniform(a, b):
    return random.uniform(a, b)

def loguniform(a, b):
    log_a = math.log(a)
    log_b = math.log(b)
    return math.exp(random.uniform(log_a, log_b))


@dataclass
class MoEParams:
    num_experts: int = 4

    moe_aux_loss: float = 0.01
    moe_kd_weight: float = 0.0
    moe_kd_temp: float = 1.0

    lambda_entropy: float = 0.05
    lambda_balance: float = 1.0

    noise_scale: float = 0.01

    # 랜덤 샘플링 모드 여부
    random: bool = False

    @classmethod
    def random_params(cls, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        return cls(
            random=True,
            num_experts=4,
            moe_aux_loss=uniform(0.011, 0.019),
            moe_kd_weight=0.0,
            moe_kd_temp=1.0,
            lambda_entropy=uniform(0.13, 0.19),
            lambda_balance=loguniform(2.5, 4.5),
            noise_scale=uniform(0.005, 0.015),
        )

    @classmethod
    def fixed(cls, **kwargs):
        """고정된 hyperparameter로 MoEParams 생성"""
        return cls(random=False, **kwargs)

    def to_dict(self) -> Dict:
        """trainer나 model.args 주입 시 쓰기 편하도록 dict로 변환"""
        return {
            "num_experts": self.num_experts,
            "moe_aux_loss": self.moe_aux_loss,
            "moe_kd_weight": self.moe_kd_weight,
            "moe_kd_temp": self.moe_kd_temp,
            "lambda_entropy": self.lambda_entropy,
            "lambda_balance": self.lambda_balance,
            "noise_scale": self.noise_scale,
        }
