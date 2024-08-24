from sagemaker.huggingface import HuggingFace
import boto3
import sagemaker
from sagemaker import Session
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
 

# Start a Sagemaker session
boto_session = boto3.Session(profile_name="Nathan")
iam = boto_session.client("iam")
role = iam.get_role(RoleName="SageMakerDev1")["Role"]["Arn"]
sagemaker_session = Session(boto_session=boto_session)
 
# define Training Job Name 
job_name = f'finetune-e5-synthetic-1-epoch'
 
 
num_re = "([0-9\\.]+)(e-?[[01][0-9])?"

metrics = [
    {"Name": 'loss', "Regex": f"'loss': {num_re}"},
]

# create the Estimator
huggingface_estimator = HuggingFace(
    sagemaker_session = sagemaker_session,
    entry_point          = 'train.py',      # train script
    source_dir           = './scripts',         # directory which includes all the files needed for training
    output_path = 's3://sagemaker-us-west-2-536930143272/acx-embeddings/',
    instance_type        = 'ml.g5.4xlarge',    # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    max_run              = 1*3*60*60,        # maximum runtime in seconds (days * hours * minutes * seconds)
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    disable_output_compression = True,        # not compress output to save training time and cost
    transformers_version = '4.36.0',
    pytorch_version      = '2.1.0',   
    py_version           = 'py310', 
    environment  = {
        "HUGGINGFACE_HUB_CACHE": "/tmp/.cache", # set env variable to cache models in /tmp
    }, 
    metric_definitions = metrics
)

huggingface_estimator.fit()