import requests
import json


if __name__ == '__main__':
    headers_dict = {"Accept": "application/fhir+json",
                    "Content-Type": "application/fhir+json",
                    "x-api-key": "8AalFGbulBqBCbdFtQXBaA9jNgOKfFN1M4iXzF50"}

    response = requests.get("https://fhir.uba1mtf6fx5u.static-test-account.isccloud.io/Patient/8112", headers=headers_dict)

    print(response.request.headers)

    type(response.json())
    # print(response.content)

    print()
