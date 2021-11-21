import requests
import json

if __name__ == '__main__':
    headers_dict = {"Accept": "application/fhir+json",
                    "Content-Type": "application/fhir+json",
                    "x-api-key": "020kStOlLF7LWx9AXjWrf6M3KMjxd68i5ruIhz4g"}
    url = "https://fhir.kt1n1r83jp32.static-test-account.isccloud.io"
    response = requests.get(f"{url}/Patient/8112/_history",
                            headers=headers_dict)

    post_res = requests.post(f"{url}/Patient",
                             headers=headers_dict,
                             json={
                                 "resourceType": "Patient",
                                 "name": [
                                     {
                                         "use": "official",
                                         "family": "Chalmers",
                                         "given": [
                                             "Peter",
                                             "James"
                                         ]
                                     }
                                 ]
                             })

    print(response.request.headers)

    type(response.json())
    print(response.content)

    print()
