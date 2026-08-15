import requests as rq

apiLink = "https://doctorsapi.com/api/doctors"

result = rq.get(apiLink, headers={
    "api-key":"hk_msup8chqb612c0d5eb728b5d6ac98fc7d370bde7a8a7745bba27470061e6942891d9beee",
    "address":"4825 S Arizona Ave, Chandler, AZ 85248",
    "radius":"30"
})

print(result.status_code, result.content)
