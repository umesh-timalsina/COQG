#!/bin/bash
# This script downloads the stanford corenlp server and runs it 
# in background. Port is 5001. Change the port number in the file below.
# The default timeout is 15000 ms.

# Download the corenlp server ( A simple wrapper with all core nlp files in a single)
wget -O corenlpserver.jar https://www.dropbox.com/s/gvjb19nmsb2bn6y/corenlpserver.jar?dl=0

# Make a data directory
mkdir -p ../data/corenlpjar

# change to the directory
mv corenlpserver.jar ../data/corenlpjar
cd ../data/corenlpjar

# Finally run the server
java -jar corenlpserver.jar 5001