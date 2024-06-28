Contributing
============

Conversations about development and issues take place in the `GitHub repository <https://github.com/PyVRP/PyVRP/>`_.
Feel free to open a new issue if you have something to discuss.


Setting up a local installation
-------------------------------

Make sure you have a reasonably modern C++ compiler.
Any recent version that supports (most of) the C++20 standard should do.
Once you have a compiler installed, you can proceed by forking the PyVRP repository from the `GitHub website <https://github.com/PyVRP/PyVRP/fork>`_.
Then, clone your new fork to some local environment:

.. code-block:: shell

   git clone https://github.com/<your username>/PyVRP.git

Now, change into the PyVRP directory, and set-up the virtual environment using ``poetry``:

.. code-block:: shell

   cd PyVRP

   pip install --upgrade poetry
   poetry install --with examples,docs,dev

This might take a few minutes, but only needs to be done once.
Now make sure everything runs smoothly, by executing the test suite:

.. code-block:: shell

   poetry run pytest

.. note::

   If you use `pre-commit <https://pre-commit.com/>`_, you can use our pre-commit configuration file to set that up too.
   Simply type:

   .. code-block:: shell

      pre-commit install

   After this completes, style and typing issues are automatically checked whenever you make a new commit to your feature branch.


Setting up Github Codespaces
----------------------------

If you are having trouble building PyVRP from source or setting up your local development environment, you can try to build PyVRP online, using `GitHub Codespaces <https://docs.github.com/en/codespaces>`_.
Github Codespaces allows you to create a development environment directly in your browser.

To launch Github Codespaces, go to the `PyVRP repository <https://github.com/PyVRP/PyVRP>`_ and click on the green button.
Select the Codespaces tab and click on the `+` icon to create a Codespaces environment.
This environment is configured with all necessary dependencies to build PyVRP.
Once the setup completes, execute the test suite to verify everything runs smoothly:

.. code-block:: shell

   poetry run pytest


Building the extensions
-----------------------

PyVRP uses a number of native Python extensions that are written in C++ for performance.
These extensions are built every time ``poetry install`` is used, but that command builds everything in release mode.
While developing, one typically wants to use debug builds.
These (and more) can be made by using the ``build_extensions.py`` script directly, as follows:

.. code-block:: shell

   poetry run python build_extensions.py

The script takes a number of command-line arguments, which you can discover using

.. code-block:: shell

   poetry run python build_extensions.py --help

We use the Meson build system to compile the C++ extensions.
Meson is configured using the ``meson.build`` file in the repository root. 
You should not have to touch this file often: all compilation is handled via the ``build_extensions.py`` script.


Debugging the extensions
------------------------

This section explains how to perform cross-debugging for mixed Python and C++ code.
We will use the `Visual Studio Code <https://code.visualstudio.com/>`_ IDE and the `Python C++ Debug <https://github.com/benibenj/vscode-pythonCpp>`_ extension.

First, build PyVRP in debug mode:

.. code-block:: shell

   poetry run python build_extensions.py --build_type debug

Create a test Python file that calls some C++ code, like so:

.. code-block:: python

   from pyvrp import Client

   Client(x=0, y=0)

Set breakpoints in ``pyvrp/cpp/ProblemData.cpp`` within the ``Client`` constructor.
Next, set-up your debugger configuration by creating the ``.vscode/launch.json`` file, with the following content:

.. code-block:: json

   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python C++ Debugger",
               "type": "pythoncpp",
               "request": "launch",
               "pythonConfig": "default",
               "cppConfig": "default (gdb) Attach"
           }
       ]
   }

Start the debugger in Visual Studio Code and step through the code.
The debugger should break at the breakpoints that you set in ``pvvrp/cpp/ProblemData.cpp``.


Committing changes
------------------

We use pull requests to develop PyVRP.
For a pull request to be accepted, you must meet the below requirements.
This greatly reduces the job of maintaining and releasing the software.

- **One branch. One feature.**
  Branches are cheap and GitHub makes it easy to merge and delete branches with a few clicks.
  Avoid the temptation to lump in a bunch of unrelated changes when working on a feature, if possible.
  This helps us keep track of what has changed when preparing a release.
- Commit messages should be clear and concise.
  This means a subject line of less than 80 characters, and, if necessary, a blank line followed by a commit message body.
- Code submissions should always include tests.
- Each function, class, method, and attribute needs to be documented using docstrings.
  We conform to the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.
- If you are adding new functionality, you need to add it to the documentation by editing (or creating) the appropriate file in ``docs/source/``.
- Make sure your documentation changes parse correctly.
  See the documentation in the ``docs/`` directory for details on how to build the documentation locally.

.. note::

   Please use the "Pull request" template on GitHub when opening a pull request.


Licensing
---------

PyVRP is licensed under the MIT license.
All code, documentation and other files added to PyVRP by contributors is licensed under this license, unless another license is explicitly specified in the source file.
For your contribution, please check that it can be included into PyVRP under the MIT license.
If you did not write the code yourself, you must ensure that the existing license is compatible and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.
Contributors keep the copyright for code they wrote and submit for inclusion to PyVRP.
