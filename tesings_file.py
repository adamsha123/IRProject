def page_iter(wiki_file):
  """ Reads a wiki dump file and create a generator that yields pages.
  Parameters:
  -----------
  wiki_file: str
    A path to wiki dump file.
  Returns:
  --------
  tuple
    containing three elements: article id, title, and body.
  """
  # open compressed bz2 dump file
  with bz2.open(wiki_file, 'rt', encoding='utf-8', errors='ignore') as f_in:
    # Create iterator for xml that yields output when tag closes
    elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))
    # Consume the first element and extract the xml namespace from it.
    # Although the raw xml has the  short tag names without namespace, i.e. it
    # has <page> tags and not <http://wwww.mediawiki.org/xml/export...:page>
    # tags, the parser reads it *with* the namespace. Therefore, it needs the
    # namespace when looking for child elements in the find function as below.
    elem = next(elems)
    m = re.match("^{(http://www\.mediawiki\.org/xml/export-.*?)}", elem.tag)
    if m is None:
        raise ValueError("Malformed MediaWiki dump")
    ns = {"ns": m.group(1)}
    page_tag = ElementTree.QName(ns['ns'], 'page').text
    # iterate over elements
    for elem in elems:
      if elem.tag == page_tag:
        # Filter out redirect and non-article pages
        if elem.find('./ns:redirect', ns) is not None or \
           elem.find('./ns:ns', ns).text != '0':
          elem.clear()
          continue
        # Extract the article wiki id
        wiki_id = elem.find('./ns:id', ns).text
        # Extract the article title into a variables called title
        # YOUR CODE HERE
        title=elem.find('./ns:title',ns).text
        # extract body
        body = elem.find('./ns:revision/ns:text', ns).text

        yield wiki_id, title, body
        elem.clear()

## Download one wikipedia file
part_url = 'https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-pages-articles-multistream15.xml-p17324603p17460152.bz2'
wiki_file = Path(part_url).name
!wget -N $part_url