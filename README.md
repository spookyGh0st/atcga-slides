# Slides for ATCGA 1 presentation

The slides here are created using [Manim](https://docs.manim.community/en/stable/index.html)
and [Manim-slides](https://eertmans.be/manim-slides/index.html).
They can be installed using `pip install manim; pip install manim-slides`
They are written for a one time presentation, so don't expect clean code or good comments.
If you want to adjust something, it's probably best to just give me a message.
I'm also open to adjust/add something if I have the time and as long it is not something 3d.

The current way to share the slides is per html.
For this a small makefile is provided, which should be straightforward and with both
`make -j 4 all` (to create the video files) and `make combine` (to create the html files) one can render the presentation to html in the settings defined in the Makefile.
If you can't use make, just look at the makefile and use the commands directly 

The slides are also available at [my website](https://spooky.moe/presentations.html).
The files are available in the corresponding [git repository](https://github.com/spookyGh0st/spookygh0st.github.io)
under p00.html, p01.html and the p00_assets and p01_assets folders.
I have also included the combined video file in this git repository in the [media/videos/](./media/videos) folder.
