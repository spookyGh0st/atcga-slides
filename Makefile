.PHONY: all combine createp0_0 createp1_0 createp1_1 createp1_2 createp1_3 combinep1 combinep0

# Set the default number of parallel jobs
# MAKEFLAGS := -j 4

all: createp1_0 createp1_1 createp1_2 createp1_3

combine: combinep0 combinep1

manim_flags := --fps 144 --resolution 1920,1080

createp02_0:
	manim ${manim_flags} p02.py p02_0

createp02_1:
	manim ${manim_flags} p02.py p02_1

createp1_0:
	manim ${manim_flags} p01.py p01_0

createp1_1:
	manim ${manim_flags} p01.py p01_1

createp1_2:
	manim ${manim_flags} p01.py p01_2

createp1_3:
	manim ${manim_flags} p01.py p01_3

createp0_0:
	manim ${manim_flags} p00.py p00_0

combinep0:
	manim-slides convert p00_0 p00.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
combinep1:
	manim-slides convert p01_0 p01_1 p01_2 p01_3 p01.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
combinep2:
	manim-slides convert p02_0 p02_1 p02.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
