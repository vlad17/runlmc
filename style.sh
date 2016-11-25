#!/bin/bash

set -e

pylint --disable=locally-disabled,fixme runlmc
