# In this file I would like to be able to connect to the FCS API and download all the Footprint Descriptors. 

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

# we use a self-signed cerificate, so must disable verification and suppress warnings
requests.packages.urllib3.disable_warnings()

CERT = "~/.certs/footprintarchive.pem"
API_URL = "https://footprint-archive.a.dmz.appbattery.akadns.net/api/v1"

class FootprintDescriptors:
    def __init__(self, api_url: str, certificate: str):
        self.api_url = api_url
        self.certificate = certificate

    def get_request(self, endpoint: str):

        raw = None
        for _ in range(3):
            raw = requests.get(
                self.api_url + endpoint, cert=self.certificate, verify=False
            )

            if raw.status_code != 200:
                logging.info(
                    f"Request to {self.api_url + endpoint} returned {raw.status_code} error."
                )
            else:
                break

        if raw == None or raw.status_code != 200:
            return None

        return json.loads(raw.text)

    def get_metros(self) -> list:
        return [data["metro"] for data in self.get_request("/scheduled/metros/")]

    def get_quarters(self) -> list:
        return [data["quarter"] for data in self.get_request("/scheduled/quarters/")]

    def get_buckets(self, metro: str, quarter: str) -> list:
        return self.get_request(f"/mapbucket/metro/{metro}/quarter/{quarter}/")

    def get_knee_for_bucket(self, metro: str, quarter: str, bucket: str):
        return self.get_request(
            f"/mapbucket/knee/metro/{metro}/quarter/{quarter}/name/{bucket}"
        )

    def get_maprules(self, metro: str, quarter: str) -> list:
        return self.get_request(
            f"/scheduled/metro/{metro}/quarter/{quarter}/content-type/LO/maprules/"
        )

    def get_knee_for_maprule(
        self, metro: str, quarter: str, network: str, maprule: str
    ):
        return self.get_request(
            f"/scheduled/knee/metro/{metro}/quarter/{quarter}/content-type/LO/network/{network}/maprule/{maprule}/"
        )

    def get_stdspace_for_maprule(self, metro: str, quarter: str, search_term: str):
        return self.get_request(
            f"/scheduled/metro/{metro}/quarter/{quarter}/content-type/LO/maprules/?search={search_term}"
        )

