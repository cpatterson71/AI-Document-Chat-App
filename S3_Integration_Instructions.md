I have updated your `app.py` file to integrate Amazon S3 for persistent storage of your document index. This means that when deployed, your `document_store.json` will be loaded from and saved to an S3 bucket, ensuring your indexed data is not lost when the server restarts or updates.

Here are the steps you need to take to configure S3 and redeploy your application:

**1. Create an Amazon S3 Bucket:**
*   If you don't already have one, create a new S3 bucket in the AWS Management Console. Choose a unique name (e.g., `your-app-name-document-store`).
*   Ensure the bucket is in the same AWS region as your Elastic Beanstalk environment.
*   For now, you can keep the default settings for the bucket, but ensure that the Elastic Beanstalk instance profile has permissions to access it (see step 3).

**2. Update `04_environment_variables.config` with your S3 Bucket Name:**
*   Open the file `C:\Users\carlp\OneDrive\Desktop\AI_Projects\AI_Document_Chat\.ebextensions\04_environment_variables.config`.
*   Add a new `option_settings` entry for `S3_BUCKET_NAME`:

    ```yaml
    option_settings:
      - namespace: aws:elasticbeanstalk:application:environment
        option_name: OPENAI_API_KEY
        value: YOUR_OPENAI_API_KEY_HERE # Make sure this is your actual key
      - namespace: aws:elasticbeanstalk:application:environment
        option_name: GOOGLE_APPLICATION_CREDENTIALS
        value: /var/app/current/Document_Store/haystack-ai-image-7d30c6a4401d.json
      - namespace: aws:elasticbeanstalk:application:environment
        option_name: S3_BUCKET_NAME
        value: your-s3-bucket-name # <--- REPLACE THIS with the name of your S3 bucket
    ```
    *(Remember to replace `your-s3-bucket-name` with the actual name of the S3 bucket you created.)*

**3. Grant S3 Permissions to your Elastic Beanstalk Instance Profile:**
*   Your EC2 instances need permission to access the S3 bucket.
*   Go to the [IAM Console](https://console.aws.amazon.com/iam/) in AWS.
*   In the left-hand navigation pane, choose **"Roles"**.
*   Find the IAM role associated with your Elastic Beanstalk instance profile. It usually has a name like `aws-elasticbeanstalk-ec2-role` or `aws-elasticbeanstalk-service-role`.
*   Click on the role, then click **"Add permissions"** > **"Attach policies"**.
*   Search for and attach the `AmazonS3FullAccess` policy (for simplicity, you can use this for now; for production, you'd want more granular permissions). Alternatively, create a custom policy that grants `s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject` permissions to your specific bucket.

**4. Redeploy your application:**
After making these changes, please terminate your current environment (if it's still unhealthy) and create a new one:
```
eb create --timeout 30
```
This will deploy the application with S3 integration and the necessary permissions.

Please let me know if you have any questions during this process or encounter any new errors.
