
# Unit Testing Process for Machine Learning

The notebooks under the **notebooks/testing** folder will help you understand the basic unit testing that could be done for different escenarios this will provide a hands-on approach to building and testing a simple machine learning pipeline using the classic Iris dataset. Consider as well as twe will be using `pytest` as outrmain library for creating the tests, if you are familiar with `unittest` lkbrary, some adjustments are needed, Here’s the explanation for each of those files:

<br>

## Testing Input Data

In this case, we will be covering two different comparisons:
- Ensuring the range of the numerical values is the same as the one we expected.
- Ensuring the data types are the same that were already defined.

1. **Initial Setup**  
   - As we explained before, we will be using pytest, and normally, you will use it only in python scripts, but for the purpose of the notebook and to make it more easy to understand, we need to install the `ipytest` library, so we can have an interactive setup.

   As explained before, we are going to create a simple model, so we need the minimum libraries required for that. Given that the iris dataset belongs from the `datasets` module from scikit-learn, we just need to make some adjustments for making it like the regular Iris dataset.

2. **Quick check in the data**  
   A simple summary in the data is performed just to take a quick overview of it.
   

3. **Setting up and running the pipeline**  
   In this part, we start setting up the different functions needed to create a Training Pipeline, in this case, we include:
   - Loading the dataset
   - Training step
   - Prediction step
   - Evaluation step
   - Execution step 
  
   Then, for execution purposes, we follow the next steps
   - We start by setting up the `SimplePipeline` class into the object *pipeline* (this is done for simplification)
   - Later, we execute the pipeline by executing the `run_pipeline` method.
   - Finally, we get the *Accuracy* by accesing into the method `get_accuracy` from the pipeline, and store it into the object *accuracy_score*.
   - At the end, we only print the metric
   
### Testing.
   - A schema, `iris_schema`, is defined to validate data input for the pipeline.
   - Two main tests are defined to validate the range of the numerical values and the data types:
     - **Range Test**: Checks if each feature’s data points are within the defined min-max range.
     - **Data Type Test**: Confirms that each feature has the correct data type as specified in `iris_schema`.

4. **Running the Tests** 
   We will use the *cell magic* `%%ipytest` which is a special command that will execute the test in a notebook cell, so we can see the result directly in the notebook with out using the terminal. And once we execute it, we will see that both test have passed.

**Note:**
Remember that using the jupyter notebook is only for visual purposes, the best practice for all of this, will having your python scripts for each tests in a folder names `test`, and each of the different tests you created needs to be named with the preffix **test**, so, if you use `pytest` or `tox`, from the command line, which is the best practice, it will be easier for the testing libraries to find all the tests you have created.

### Failure escenarios

As we are testing the input in our data, the only thing included here, is a change in the assertion values we are testing:
- The maximum value is lower than zero.
- The minimum value is greater than 1000
- And we the test to ensure that the data types in the test are different to those we have set in the schema.

<br>

## Testing Feature Engineering Process.

This notebook *Testing_Feature_Engineering_Process.ipynb* will build up into the previous one, so you will find many steps that are similar to the *Testing_Input_Data.ipynb*. The main difference is that a new class `PipelineWithFeatureEngineering` is included. This new class contains the following:

- It inherits functionality from the `SimplePipeline` class defined before.
- It calls the `StandardScaler` method from *scikit-learn* and fit it into the training dataset.
- It applies the scaler into the test and training dataset
- The predictions are created using the *scaled_input_data*

Now, rather than running the `SimplePipeline` class, and importing it's methods, we run the `PipelineWithFeatureEngineering` class and follows the same steps.


### Testing.

For the tests in this notebook, the main idea is to ensure that, after applying the feature engineering process, the new mean is lower than the original mean in our features and close to zero, and that the SD is close to one as well.

### Failure scenarios.

In this case, we are considering two different scenarios:
- The first one, aims to check if the new mean obtained after applying the `StandardScaler` is grater than the original mean and changing the value in `isclose` in the SD, so, this one only pass if we got an exact value.
- The second scenario, only adds variation in the dataset, then run the pipeline, and check how near is with the original tolerance. 

<br>

## Testing Model Quality.

The purpose of this notebook (*Testing_Feature_Engineering_Process.ipynb*) is to showcase some comparissons between the metric chosen for evaluating the performance of the model we created, in this case, is *Accuray*, but it would be any metric you like.

### Testing

First, a benchmark is setup, so we can compare how our model makes it against this benchmark. If you look closer to the test crrated for this, it's name is *test_accuracy_higher_than_benchmark* and as you can see, the logic expression for the assertion is:

$$ actual_accuracy > benchmark_accuracy $$

The results of the *actual_accuracy* which is build with the `SimplePipeline` class are: 0.9693877551020408; while from the Benchmark, the results are: 0.32653061224489793

**Note:**
1. Those results may vary if you change some of the arguments (random_state and test_size) in the `SimplePipeline` class. 
2. If you want to make this test fail, it would be enough to reverse the logical operator (from > to <).

In the next evaluation, we are comparing the performance of both models, the one we created with the `SimplePipeline` class and the other one build with the `PipelineWithFeatureEngineering` class, so the idea is to know if the new model performs better than the first one.

$$ v2_accuracy >= v1_accuracy $$

And in this case, the test failed because the accuracy of the model with the `StandarScaler` is 0.9591836734693877 while the accuracy of the simple model is 0.9693877551020408.

<br>

## Testing Model Settings.

In this file, we use the `SimplePipeline` class, just to explain how you could make changes and testing for different configuration details (hyperparameters, random seed, folds, performance metrics, etc.).

We create a new class `PipelineWithConfig` that inherits the main functionality from the `SimplePipeline`, this new class, uses Dependency Injection which is a design pattern to manage dependencies between different components, it is more common use for testing, modularity and reproducibility.

In this new class, by injecting the *config*, we make it easy to adjust or swap the configuration without modifying the `PipelineWithConfig` class. For instance, you could use different configurations for testing, tuning, or deploying the model with alternative solvers or multi-class settings.

And in the testing setup, we asked the test to run using the defined solver in the *config*, which is the `lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) which tends to be the most common solver given it's balance between computing performance and accuracy. But, the only solver available is `newton-cg` (Newton-Conjugate Gradient), so, in this case, the test will fail.

To make this test succeed, we only need to include the `lbfgs` in the **ENABLE_MODEL_SOLVERS** object.