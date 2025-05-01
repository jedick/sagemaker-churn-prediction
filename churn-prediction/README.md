# Churn Prediction with SageMaker Pipelines

## By Jeffrey Dick on 2025-05-01

There are two sources for this demo:

- The AWS blog post [Build, tune, and deploy an end-to-end churn prediction model using Amazon SageMaker Pipelines](https://aws.amazon.com/blogs/machine-learning/build-tune-and-deploy-an-end-to-end-churn-prediction-model-using-amazon-sagemaker-pipelines/).
- The AWS sample code in [this GitHub repository](https://github.com/aws-samples/customer-churn-sagemaker-pipelines-sample)

Organization:

- `Churn_Prediction_Interactive.ipynb`: Interactive data preprocessing, loading data splits into S3, and hyperparameter optimization
- `Churn_Prediction_Pipeline.ipynb`: Orchestrates all model steps using SageMaker pipeline

## Introduction

- Predicting whether a customer stops paying (or churns) is an important ability for businesses
- The end-to-end analytics workflow involves data preparation, experimenting with baseline models, hyperparameter optimization, and model registration
- Amazon SageMaker is designed to provide all ML development steps: build, train, and deploy models
- SageMaker Pipelines automates model building and support CI/CD
- SageMaker Clarify pinpoints biases and generates explanations for stakeholders'

## Aims and challenges

This demo:

- Puts together these technologies to implement a churn prediction model
- Uses updated Python libraries and SageMaker features compared to the blog post (dated from 2021)
- Deals with access policies for S3 buckets

## Data preparation

- Download the data from [Customer Retention Retail](https://www.kaggle.com/datasets/uttamp/store-data) on Kaggle
- The unzipped file name is `storedata_total.xlsx`
- Use a spreadsheet program to convert the file to CSV format
- Create an S3 bucket named `churn-prediction-sagemaker-demo`
- In the bucket, create a directory named `demo`
- Upload `storedata_total.csv` to the `data` directory of the bucket

## S3 bucket policies

After creating the S3 bucket, we need to grant access to an IAM user in the same account

- See [Examples of Amazon S3 bucket policies](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies.html#example-bucket-policies-folders)
- This policy allows all actions in order to list, read, and upload files and directories
- **Be aware of the security implications of this policy**
- Add this example bucket policy (replace the sample user ARN with the notebook execution role printed below)

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowAllS3Actions",
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::111122223333:user/JohnDoe"
                ]
            },
            "Action": ["s3:*"],
            "Resource": ["arn:aws:s3:::churn-prediction-sagemaker-demo/*"]
        }
    ]
}
```

This is how to get the [execution role for the SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html):

```python
from sagemaker import get_execution_role

role = get_execution_role()
print(role)
```

*Edit*: Adding the above ARN to the S3 bucket policy didn't allow access from the SageMaker notebook.
Instead, I had to add this ARN to the policy (using the [STS.Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts.html) method from Boto3):

```python
import boto3

client = boto3.client('sts')
identity = client.get_caller_identity()
print(identity["Arn"])
```

## Data preprocessing

- Here we use Boto3 to access data from S3
    - The [sample notebook](https://github.com/aws-samples/customer-churn-sagemaker-pipelines-sample) uses [s3fs](https://github.com/fsspec/s3fs)
    - However, s3fs is not an official AWS product, and there are [issues with using it in SageMaker notebooks](https://repost.aws/questions/QUqm1CyclTQzinmwXZ6OiFLw/installing-s3fs-causes-errors-in-jupyterlab-on-sagemaker)
    - Boto3 is the AWS SDK: [API documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- The result is stored in the `storedata` object for further processing
- *Note*: The raw data has 30801 rows, while the processed data has 9307 rows

## SageMaker setup

I got this error when first trying to run a training job:

```
ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateHyperParameterTuningJob operation: The account-level service limit 'ml.m4.xlarge for training job usage' is 0 Instances, with current utilization of 0 Instances and a request delta of 2 Instances. Please use AWS Service Quotas to request an increase for this quota. If AWS Service Quotas is not available, contact AWS support to request an increase for this quota
```                                                       

- I requested a quota increase in the AWS Console (see [these instructions](https://repost.aws/knowledge-center/sagemaker-resource-limit-exceeded-error)) ...
- A support case was opened at 2:40 p.m. on April 29.
