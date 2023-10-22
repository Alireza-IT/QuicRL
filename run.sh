

for i in {1..500}; do python3 examples/http3_client.py --ca-certs tests/pycacert.pem https://localhost:4433/pic1.png ; done
