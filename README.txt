READ ME

This project is coded using Python.

This folder is broken into several subfolders.

To run the entire script in an automated workflow: 

Open your Python IDE and in the terminal run the command:

pip install -r requirements.txt

This will check your IDE for all necessary packages to run the processes.

Next, open the Main_process_flow script file.

Where it says "YOUR-DIRECTORY", replace this with the path of the "workflow" folder on your machine.

Once you've done this you can run the script and it will run all of the necessary sub-processes.

If there are errors that stop the script you have 2 options:

1.	Remove the offending script from the "for script in ["...." line and run again without that one. You can run that individual script as a notebook from the notebooks folder.
2.	Run all of the scripts as individual notebooks from the notebooks folder. Maintain the order that they are given in the "for script in[.." section.