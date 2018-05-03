
#Libraries
import pandas as pd
import numpy as np

class XmlToDataframe():
    """
    Class to convert Web of Science XML query data to Pandas dataframe.
    """
    
    def __init__(self, xmlData):
        """
        Initialize class and load xmlData to convert to dataframe.
        
        Parameters
        ----------
        xmlData: xml-like
            Xml hierarchical tree created using WoS API (see wos_api.py)
        """
        self.xmlData = xmlData
            
    def get_node_text(self, node):
        """
        Returns text of node if available, otherwise None.
        
        Parameters
        ----------
        node: xml element
            XML tree node that contains text.
        
        Returns
        -------
        text from input node
        """
        return node.text if node is not None else np.nan

    def get_node_attrib(self, node, key):
        """
        Returns value of key if attribute is a dictionary, otherwise None.
        
        Parameters
        ----------
        node: xml element
            XML tree node that contains attribute(s).
        
        Returns
        -------
        attribute(s) from input node
        """
        return node.attrib[key] if node is not None else np.nan
    
    def convert_to_dataframe(self):
        """
        Runs through default tags to parse and convert to pandas dataframe.
        Uses get_node_text and get_node_attrib functions above to grab xml information.
        
        Returns
        -------
        df: pandas dataframe with each row as a document in corpus
        """

        #Initialize list used to create dataframe at end
        listParse = [] # list that will have dictionary appended for each record

        for node in self.xmlData.getroot(): #loop over each record (root in WoS xml results)
            dictParse = {} #dictionary for single record
            
            # WoS Accession Number
            dictParse['UID'] = self.get_node_text(node.find('UID'))
    
            # Authors - add author names to string; do for both display_name and wos_name
            authors = ''
            for ele in node.findall('static_data/summary/names/name/display_name'):
                authors = authors + str(self.get_node_text(ele)) + '; '
            dictParse['Authors'] = authors[:-2]
    
            authorsWos = ''
            for ele in node.findall('static_data/summary/names/name/wos_standard'):
                authorsWos = authorsWos + str(self.get_node_text(ele)) + '; '
            dictParse['Authors WoS'] = authorsWos[:-2]
    
            # Addresses: create string of all full addresses
            addresses = ''
            for ele in node.findall("static_data/fullrecord_metadata/addresses/" \
                                    "address_name/address_spec/full_address"):
                addresses = addresses + str(self.get_node_text(ele)) + '; '
            dictParse['Addresses'] = addresses[:-2]
    
            # Title - loop through title tag for source and title
            for ele in node.iter(tag="title"):
                if ele.attrib['type'] == 'item':
                    dictParse['Title'] = self.get_node_text(ele)
                elif ele.attrib['type'] == 'source':
                    dictParse['Source'] = self.get_node_text(ele)
    
            # Publication Year, Date, Type
            dictParse['Publication Type'] = self.get_node_attrib(node.find('static_data/' \
                                                                           'summary/pub_info'),
                                                                 'pubtype')
            docType = ''
            for ele in node.findall('static_data/fullrecord_metadata/normalized_doctypes/doctype'):
                docType = docType + str(self.get_node_text(ele)) + '; '
            dictParse['Normalized Type'] = docType[:-2]
            dictParse['Publication Year'] = self.get_node_attrib(node.find('static_data/' \
                                                                           'summary/pub_info'),
                                                                 'pubyear')
            dictParse['Publication Date'] = self.get_node_attrib(node.find('static_data/' \
                                                                           'summary/pub_info'),
                                                                 'sortdate')    
    
            # Technology category areas - both Research Areas of WoS tags (traditional)
            researchArea = ''
            wosArea = ''
            for ele in node.findall('static_data/fullrecord_metadata/category_info/subjects/subject'):
                if ele.attrib['ascatype'] == 'extended':
                    researchArea = researchArea + str(self.get_node_text(ele)) + '; '
            for ele in node.findall('static_data/fullrecord_metadata/category_info/subjects/subject'):
                if ele.attrib['ascatype'] == 'traditional':
                    wosArea = wosArea + str(self.get_node_text(ele)) + '; '
            dictParse['Category: Research'] = researchArea[:-2]
            dictParse['Category: WoS'] = wosArea[:-2]

            # Author Keywords
            keywordsAuth = ''
            for ele in node.findall('static_data/fullrecord_metadata/keywords/*'):
                keywordsAuth = keywordsAuth + str(self.get_node_text(ele)) + '; '
            dictParse['Keywords Author'] = keywordsAuth[:-2]
            
            # Keywords Plus
            keywordsPlus = ''
            for ele in node.findall('static_data/item/keywords_plus/keyword'):
                keywordsPlus = keywordsPlus + str(self.get_node_text(ele)) + '; '
            dictParse['Keywords Plus'] = keywordsPlus[:-2]
    
            # Citations
            dictParse['Times Cited'] = self.get_node_attrib(node.find('dynamic_data/' \
                                                                      'citation_related/' \
                                                                      'tc_list/silo_tc'),
                                                            'local_count')
    
            # Identifiers - DOI and ISSN
            for ele in node.findall('dynamic_data/cluster_related/identifiers/'):
                if ele.attrib['type'] == 'issn':
                    dictParse['ISSN'] = self.get_node_attrib(ele, 'value')
                if ele.attrib['type'] == 'doi':
                    dictParse['DOI'] = self.get_node_attrib(ele, 'value')
            
            # Abstract
            dictParse['Abstract'] = self.get_node_text(node.find('static_data/' \
                                                                 'fullrecord_metadata/' \
                                                                 'abstracts/abstract/' \
                                                                 'abstract_text/p'))
            
            # Funding Text
            dictParse['Funding Text'] = self.get_node_text(node.find('static_data/' \
                                                                     'fullrecord_metadata/' \
                                                                     'fund_ack/fund_text/p'))
            
            # Append dictionary to list
            listParse.append(dictParse)

        # Convert columns to correct types
        df = pd.DataFrame(listParse)
        df['Times Cited'] = df['Times Cited'].astype(int)
        df['Publication Date'] = pd.to_datetime(df['Publication Date'])
        df['Publication Year'] = pd.to_datetime(df['Publication Year'])
        
        # After loop through all records create dataframe from list
        return df
        