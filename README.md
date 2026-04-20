---
license: apache-2.0
metrics:
- mse
- mae
tags:
- time series
- forecasting
- foundation models
- pretrained models
- generative models
- time series foundation models
library_name: transformers
---

# SymTime

~~~python
from transformers import AutoModel
model = AutoModel.from_pretrained("FlowVortex/SymTime", trust_remote_code=True)
~~~