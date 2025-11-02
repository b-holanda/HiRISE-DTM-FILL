AWS_ACCOUNT_ID="109762776171"

AWS_REGION="us-east-1"

ECR_REPOSITORY="marsfill-training-env"

set -e

if ! command -v aws &> /dev/null; then
    echo "ERRO: AWS CLI não encontrado. Por favor, instale-o e rode 'aws configure'."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERRO: Docker não encontrado. Por favor, inicie o serviço Docker."
    exit 1
fi

echo "✅ Pré-requisitos verificados."

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

IMAGE_TAG=$(git rev-parse --short HEAD)
IMAGE_URI_SHA="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
IMAGE_URI_LATEST="${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"

echo "-----------------------------------------"
echo "Construindo Imagem: ${IMAGE_URI_LATEST}"
echo "Tag do Commit: ${IMAGE_TAG}"
echo "-----------------------------------------"

echo "🔑 Autenticando Docker no ECR Privado..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
echo "Login no ECR Privado concluído."

echo "🔑 Autenticando Docker no ECR Público (para imagem base)..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
echo "Login no ECR Público concluído."

echo "🐳 Construindo a imagem (usando Dockerfile)..."
docker build -t $IMAGE_URI_SHA -f Dockerfile .
echo "Build concluído."

echo "🏷️  Adicionando tag 'latest'..."
docker tag $IMAGE_URI_SHA $IMAGE_URI_LATEST
echo "Tag 'latest' adicionada."

echo "🚀 Enviando tag SHA ($IMAGE_TAG)..."
docker push $IMAGE_URI_SHA

echo "🚀 Enviando tag 'latest'..."
docker push $IMAGE_URI_LATEST

echo "---"
echo "✅ SUCESSO!"
echo "Sua imagem está pronta no ECR:"
echo "   ${IMAGE_URI_LATEST}"
echo "---"
