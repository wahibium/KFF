%%Executes the program at location with arguments contained in agr 

function succes=RunSimul(location,port,program,arg)


tempstring=['ssh -p ' num2str(port) ' '  location program ' ' arg ];


tempstring

system(tempstring);