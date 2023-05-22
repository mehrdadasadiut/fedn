#!/bin/bash
set -e
pwd
client/entrypoint init_seed

# Make compute package
tar -czvf package.tgz client
