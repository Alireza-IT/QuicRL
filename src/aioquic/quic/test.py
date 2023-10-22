from aioquic.quic.recovery import QuicCongestionControl
from examples.http3_client import HttpClient

qc = QuicCongestionControl()
qc.reduction = 0.1
print(qc.reduction)

hclient = HttpClient()

result = hclient.http_event_received()

print(result)