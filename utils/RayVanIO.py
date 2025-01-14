import paho.mqtt.client as mqtt
import json
class RayvanIOConnector:
    """RayvanIOConnector"""
    def __init__(self,host="",username="",password="",topic=""):
        self.mqtt_client=mqtt.Client(client_id="XVhsasfgas", clean_session=True, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
        self.host=host
        self.topic=topic
        self.port=1883
        self.username=username
        self.password=password

    def connect(self):
        """connecting to the server."""
        self.mqtt_client.username_pw_set(self.username,self.password)
        self.mqtt_client.connect(self.host,self.port)
        self.mqtt_client.loop_start()

    def send_telemetry(self,data):
        """Sending data as telementry"""
        self.mqtt_client.publish(self.topic,json.dumps(data))

    def disconnect(self):
        """stopping the client"""
        self.mqtt_client.loop_stop()