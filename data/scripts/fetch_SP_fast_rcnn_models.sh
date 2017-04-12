#!/usr/bin/env bash


echo "Downloading models..."


if [ -f "SP_models.zip" ]; then
   rm SP_models.zip
fi

wget https://www.dropbox.com/s/1yg85qxfvw7akkx/SP_models.zip

echo "Unzipping..."
unzip SP_models.zip && rm -f SP_models.zip

echo "Done."
