Before filing an issue, please make sure to tick the following boxes.

- [ ] Make sure your issue hasn't been filed already. Use GitHub search or manually check the [existing issues](https://github.com/maxpumperla/hyperas/issues), also the closed ones. Also, make sure to check the FAQ section of our [readme](https://github.com/maxpumperla/hyperas/blob/master/README.md).

- [ ] Install latest hyperas from GitHub:
pip install git+git://github.com/maxpumperla/hyperas.git

- [ ] Install latest hyperopt from GitHub:
pip install git+git://github.com/hyperopt/hyperopt.git

- [ ] We have continuous integration running with Travis and make sure the build stays "green". If, after installing test utilities with `pip install pytest pytest-cov pep8 pytest-pep8`, you can't successfully run `python -m pytest` there's very likely a problem on your side that should be addressed before creating an issue.

- [ ] Create a gist containing your complete script, or a minimal version of it, that can be used to reproduce your issue. Also, add your _full stack trace_ to that gist. In many cases your error message is enough to at least give some guidance.
