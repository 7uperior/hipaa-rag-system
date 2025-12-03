docker compose up -d --build
docker compose exec backend python scripts/run_etl.py

vs@vspc:~/career/hipaa$ curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What is minimum necessary?"}' | jq
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--    100    38    0     0  100    38      0     31  0:00:01  0:00:01 --:--:--    100    38    0     0  100    38      0     17  0:00:02  0:00:02 --:--:--    100    38    0     0  100    38      0     11  0:00:03  0:00:03 --:--:--    100    38    0     0  100    38      0      9  0:00:04  0:00:04 --:--:--    100    38    0     0  100    38      0      7  0:00:05  0:00:05 --:--:--    100    38    0     0  100    38      0      6  0:00:06  0:00:06 --:--:--    100    38    0     0  100    38      0      5  0:00:07  0:00:07 --:--:--    100    38    0     0  100    38      0      4  0:00:09  0:00:08  0:00:01    100  1133  100  1095  100    38    126      4  0:00:09  0:00:08  0:00:01   2100  1133  100  1095  100    38    126      4  0:00:09  0:00:08  0:00:01   317
{
  "answer": "Minimum necessary is a key principle under HIPAA that requires covered entities and business associates to limit the use, disclosure, and requests for protected health information to the minimum necessary to accomplish the intended purpose (§ 164.502(b)(1)). This principle is reflected in different aspects of HIPAA regulations:\n\n1. Minimum necessary requests for protected health information are addressed in § 164.514(i)(4).\n2. Minimum necessary disclosures of protected health information are discussed in § 164.514(i)(3).\n3. Minimum necessary uses of protected health information are outlined in § 164.514(c)(2).\n\nOverall, covered entities and business associates must implement policies and procedures to ensure that only the minimum necessary information is used, disclosed, or requested to carry out specific functions or activities (§ 164.502(b)(1)). Failure to adhere to the minimum necessary standard may result in violations of HIPAA regulations and potential penalties.",
  "sources": [
    "164.514(i)(4)",
    "164.514(i)(3)",
    "164.514(c)(2)",
    "164.502(b)(1)",
    "162.910(b)"
  ]
}


docker compose exec backend python -m scripts.run_evaluation
