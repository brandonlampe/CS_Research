import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def permDoc(tname):
    """ docstring
    """

    doc = SimpleDocTemplate(str(tname)+str(.pdf), 
    	pagesize=letter,rightMargin=1*inch, 
    						leftMargin=1*inch, topMargin=1*inch, 
    						bottomMargin=0.25*inch)

	Story = []
	FMT_TIME = time.ctime()

	Story.append(Space(1, 12))