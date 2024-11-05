# Pre-release Tests

1. Create a new environment. For instance, using conda:
```
conda create --name fresh_env python=3.11
```
2. Activate the new environment
```
conda activate fresh_env
```
3. Clone the dev version of SpeechBrain
https://github.com/speechbrain/speechbrain

4. Install the extra-dependencies
```
cd speechbrain
pip install -r requirements.txt
```
5. Install SpeechBrain
```
pip install -e .
```
6. Install all recipe extra-dependencies (check for latest/fixed versions)
```
find recipes | grep extra | xargs cat | sort -u | grep -v \# | xargs -I {} pip install {}
pip install fairseq
conda install 'ffmpeg<4.4'
```
7. Update the PERFORMANCE.md file:
```
python tools/readme_builder.py --recipe_info_dir tests/recipes/ --output_file PERFORMANCE.md
```
Remember to push it.

8. Run the basic tests by typing:
```
pytest
```
9. Run load yaml test:
```
tests/.run-load-yaml-tests.sh
```
10. Run recipe tests
```
tests/.run-recipe-tests.sh
```
11. Make sure all HuggingFace repos are working
```
tests/.run-HF-checks.sh
```
12. Make sure all HuggingFace API Interfaces are up to date and working (see [here](#huggingface-api-testing)])
13. Check URLs
```
tests/.run-url-checks.sh
```

Make sure all the tests are passing. Also, make sure to check that the tutorials are working (we might set up an automatic test for that as well in the future).

# HuggingFace API testing

API testing cannot be automated within SpeechBrain, however, it is already done within HuggingFace repository!
Steps to test them are:
1. Go to the [api-inference-community](https://github.com/huggingface/api-inference-community) and clone it.
2. Create a new conda environment and do ```pip install -r docker_images/speechbrain/requirements.txt```
3. Then ```pip install -r requirements.txt``` and ```pip install -e .``` to install the api-inference-community package
4. If not already installed, please install ffmpeg with  ```conda install -c conda-forge ffmpeg```
5. Go to the [test file](https://github.com/huggingface/api-inference-community/blob/main/docker_images/speechbrain/tests/test_api.py) and make sure that all the models that you want to test are here. Ideally we just want one model per interface.
6. Run ```pytest -sv --rootdir docker_images/speechbrain/ docker_images/speechbrain/``` and make sure that all tests are passing.

if tests fail, it is most likely because one interface is missing, hence follow the next steps.

## Adding an interface to the HuggingFace API.

1. Go to the [api-inference-community](https://github.com/huggingface/api-inference-community) and clone it.
2. Add the interface name to the [ModelType file](https://github.com/huggingface/api-inference-community/blob/main/docker_images/speechbrain/app/common.py). This correspond to the ALL CAPITALIZED variables.
3. If your interface can be derived from one of the existing pipelines in [here](https://github.com/huggingface/api-inference-community/tree/main/docker_images/speechbrain/app/pipelines), simply go to the good one, for instance [automatic-speech-recognition](https://github.com/huggingface/api-inference-community/blob/main/docker_images/speechbrain/app/pipelines/automatic_speech_recognition.py) and add your new interface *if statement*.
4. If your interface cannot be derived from an existing pipeline, then you will need to create a new file [here](https://github.com/huggingface/api-inference-community/tree/main/docker_images/speechbrain/app/pipelines) and contact the HuggingFace team to move forward (see our HuggingFace Slack channel).
5. Test your changes (see previous section).
6. Once done, simply do a PR to the api-inference-community!

# Maintainer checks for releases

Up until here, all the above madness should have settled.
Commit logs outline what happened; features are summarized.

_Note: a good point to check https://speechbrain.github.io/ is up-to-date._

The task at hand is:
* change the version number;
* compile a change log, and
* release the latest version on PyPI.

Another CI/CD lifecycle begins.
