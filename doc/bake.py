import sys
import re


if __name__ == '__main__':
  content = open(sys.argv[1]).read()
  renders = re.findall('<img.*/>', content)
  for render in renders:
    svg = render[render.index('svgs/') : render.index('?invert_in_darkmode')]
    print('![]({})'.format(svg))
    content = content.replace(render, '![]({})'.format(svg))

  open(sys.argv[1], 'w').write(content)
