import urllib.request
import tarfile
import ssl

# SSL 인증서 확인 건너뛰기 설정
ssl._create_default_https_context = ssl._create_unverified_context

# 다운로드
url = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000375/data/data.tar.gz"
urllib.request.urlretrieve(url, "data.tar.gz")

# 압축 해제
with tarfile.open("data.tar.gz", "r:gz") as tar:
    tar.extractall()

print("완료!")
