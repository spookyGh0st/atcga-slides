.PHONY: all combine createp0_0 createp1_0 createp1_1 createp1_2 createp1_3 combinep1 combinep0

# Set the default number of parallel jobs
# MAKEFLAGS := -j 4

all: createp02 createp03

combine: combinep02 combinep03

manim_flags := -qh # --fps 144 --resolution 1920,1080

PHONY createp02: createp02_0 createp02_1
createp02_0:
	sleep 0; manim ${manim_flags} p02.py p02_0
createp02_1:
	sleep 1; manim ${manim_flags} p02.py p02_1

PHONY createp03: createp03_0 createp03_1 createp03_3
createp03_0:
	sleep 0; manim ${manim_flags} p03.py p03_0
createp03_1:
	sleep 1; manim ${manim_flags} p03.py p03_1
createp03_2:
	sleep 2; manim ${manim_flags} p03.py p03_2
createp03_3:
	sleep 2; manim ${manim_flags} p03.py p03_3


PHONY createp01: createp01_0 createp01_1 createp01_2 createp01_3
createp01_0:
	sleep 0; manim ${manim_flags} p01.py p01_0
createp01_1:
	sleep 1; manim ${manim_flags} p01.py p01_1
createp01_2:
	sleep 2; manim ${manim_flags} p01.py p01_2
createp01_3:
	sleep 3; manim ${manim_flags} p01.py p01_3

createp00:
	manim ${manim_flags} p00.py p00_0

combinep0:
	sleep 0; manim-slides convert p00_0 p00.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
combinep1:
	sleep 1; manim-slides convert p01_0 p01_1 p01_2 p01_3 p01.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
combinep02:
	sleep 2; manim-slides convert p02_0 p02_1 p02.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
combinep03:
	sleep 3; manim-slides convert p03_0 p03_1 p03_2 p03.html --use-template revealjs_template.html -ccontrols=true -ccontrols_layout=edges -cslide_number=true
