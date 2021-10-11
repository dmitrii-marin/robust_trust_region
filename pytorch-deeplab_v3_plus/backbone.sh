


[ -n "$BACKBONE" ] ||  BACKBONE=mobilenet

if [ "$BACKBONE" == mobilenet ]; then
	CP_PREFIX=mn
elif
   [ "$BACKBONE" == resnet ]; then
        CP_PREFIX=rn
fi

