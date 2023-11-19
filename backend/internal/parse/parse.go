package parse

import (
	_ "fmt"
	"log"
	// "fmt"
	"github.com/antchfx/htmlquery"
	"io/ioutil"
	_ "encoding/json"
	"backend/pb"
)

//
//
// Кароче еба там куча конкатинаций и повторений (сами чекните)
// Нужно почистить
// 8:---D

const (
	ඞ = "ඞ"
)

// type Resume struct {
// 	ID string `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
// 	Location string `protobuf:"bytes,2,opt,name=location,proto3" json:"location,omitempty"`
// 	Title string `protobuf:"bytes,3,opt,name=title,proto3" json:"title,omitempty"`
// 	Specs string `protobuf:"bytes,4,opt,name=specs,proto3" json:"specs,omitempty"`
// 	Salary string `protobuf:"bytes,5,opt,name=salary,proto3" json:"salary,omitempty"`
// 	Zgr []string `protobuf:"bytes,6,rep,name=zgr,proto3" json:"zgr,omitempty"`
// 	Exp_title string `protobuf:"bytes,7,opt,name=exp_title,proto3" json:"exp_title,omitempty"`
// 	Experience []string `protobuf:"bytes,8,rep,name=experience,proto3" json:"experience,omitempty"`
// 	Skills []string `protobuf:"bytes,9,rep,name=skills,proto3" json:"skills,omitempty"`
// 	Driver string `protobuf:"bytes,10,opt,name=driver,proto3" json:"driver,omitempty"`
// 	About string `protobuf:"bytes,11,opt,name=about,proto3" json:"about,omitempty"`
// 	Recomend string `protobuf:"bytes,12,opt,name=recomend,proto3" json:"recomend,omitempty"`
// 	Portfolio string `protobuf:"bytes,13,opt,name=portfolio,proto3" json:"portfolio,omitempty"`
// 	Education []string `protobuf:"bytes,14,rep,name=education,proto3" json:"education,omitempty"`
// 	Langs string `protobuf:"bytes,15,opt,name=langs,proto3" json:"langs,omitempty"`
// 	AdditionalEducation []string `protobuf:"bytes,16,rep,name=additional_education,proto3" json:"additional_education,omitempty"`
// 	Tests []string `protobuf:"bytes,17,rep,name=tests,proto3" json:"tests,omitempty"`
// 	Certificates string `protobuf:"bytes,18,opt,name=certificates,proto3" json:"certificates,omitempty"`
// 	AdditionalInfo string `protobuf:"bytes,19,opt,name=additional_info,proto3" json:"additional_info,omitempty"`
// }

// type Resumes struct {
// 	Resumes []Resume `protobuf:"bytes,1,rep,name=resumes,proto3" json:"resumes,omitempty"`
// }


func Parse() []*pb.Resume {
	files, err := ioutil.ReadDir("data")
	if err != nil {
		log.Fatal(err)
	}

	var resumes []*pb.Resume = make([]*pb.Resume, len(files))

	for _, f := range files {
		filePath := "./data/" + f.Name()
		doc, err := htmlquery.LoadDoc(filePath)
		if err != nil {
			log.Fatal(err)
		}

		resume := pb.Resume{}

		resume.ID = f.Name()
		// location
		for _, n := range htmlquery.Find(doc, "//div[@class='resume-header']//div[@class='resume-header-title']") {
			resume.Location = htmlquery.InnerText(n.LastChild.FirstChild)
		}

		// title
		for _, n := range htmlquery.Find(doc, "//span[@data-qa='resume-block-title-position']") {
			resume.Title = htmlquery.InnerText(n)
		}

		// specs
		for _, n := range htmlquery.Find(doc, "//li[@data-qa='resume-block-position-specialization']") {
			resume.Specs = htmlquery.InnerText(n)
		}

		// salary
		for _, n := range htmlquery.Find(doc, "//span[@class='resume-block__salary']") {
			resume.Salary = htmlquery.InnerText(n)
		}
	
		// zgr
		res := htmlquery.Find(doc, "//div[@class='resume-block-item-gap']")
		if len(res) > 0 {
			zgr := htmlquery.Find(res[0], "//p")
			for _, p := range zgr {
				resume.Zgr = append(resume.Zgr, htmlquery.InnerText(p))
			}
		}


		// expanding....
		for _, n := range htmlquery.Find(doc, "//div[@data-qa='resume-block-experience']") {
			// find_all('div', {'class':'bloko-columns-row'})
			list := htmlquery.Find(n, "//div[@class='bloko-columns-row']")
			// exp_title
			if len(list) > 0 {
				resume.ExpTitle = htmlquery.InnerText(list[0])
			}


			for i := 1; i < len(list); i++ {
				var exp string = ""
				
				// time_
				timeq := htmlquery.Find(list[i], "//div[@class='bloko-column bloko-column_xs-4 bloko-column_s-2 bloko-column_m-2 bloko-column_l-2']")
				if len(timeq) > 0 {
					exp += htmlquery.InnerText(timeq[0]) + ඞ
				}

				// name_
				nameq := htmlquery.Find(list[i], "//div[@class='bloko-text bloko-text_strong']")
				if len(nameq) > 0 {
					exp += htmlquery.InnerText(nameq[0]) + ඞ
				}

				// position_
				posq := htmlquery.Find(list[i], "//div[@data-qa='resume-block-experience-position']")
				if len(posq) > 0 {
					exp += htmlquery.InnerText(posq[0]) + ඞ
				}


				// place_
				placeq := htmlquery.Find(list[i], "//p")
				if len(placeq) > 0 {
					exp += htmlquery.InnerText(placeq[0]) + ඞ
				}

				// industry_ нет таких данных Ж(
				indq := htmlquery.Find(list[i], "//div[@data-qa='resume-block__experience-industries resume-block_no-print']")
				if len(indq) > 0 {
					exp += htmlquery.InnerText(indq[0]) + ඞ
				}

				// description_
				descq := htmlquery.Find(list[i], "//div[@data-qa='resume-block-experience-description']")
				if len(descq) > 0 {
					exp += htmlquery.InnerText(descq[0])
				}


				resume.Experience = append(resume.Experience, exp)
			}
		}

		// skills = soup.select_one('div[class="bloko-tag-list"]')
		skills := htmlquery.Find(doc, "//div[@class=\"bloko-tag-list\"]")
		// var skills_arr []string = make([]string, len(skills))
		for _, n := range skills {
			for _, p := range htmlquery.Find(n, "//div") {
				resume.Skills = append(resume.Skills, htmlquery.InnerText(p))
			}
		}

		// driver 
		driver := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-driver-experience\"]")
		if len(driver) > 0 {
			resume.Driver = htmlquery.InnerText(driver[0])
		}

		// about 
		about := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-skills\"]")
		if len(about) > 0 {
			resume.About = htmlquery.InnerText(about[0])
		}

		//recomend div[data-qa="recommendation-item-title"]
		recomend := htmlquery.Find(doc, "//div[@data-qa=\"recommendation-item-title\"]")
		if len(recomend) > 0 {
			resume.Recomend = htmlquery.InnerText(recomend[0])
		}

		// portfolio div[data-qa="resume-block-portfolio"]
		portfolio := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-portfolio\"]")
		if len(portfolio) > 0 {
			resume.Portfolio = htmlquery.InnerText(portfolio[0])
		}

		// edus div[data-qa="resume-block-education"]
		edus := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-education\"]//div[@class=\"bloko-columns-row\"]")
		for i := 2; i < len(edus); i++ {
			ed := ""
			years := htmlquery.Find(edus[i], "//div[@class=\"bloko-column bloko-column_xs-4 bloko-column_s-2 bloko-column_m-2 bloko-column_l-2\"]")
			name := htmlquery.Find(edus[i], "//div[@data-qa=\"resume-block-education-name\"]")
			place := htmlquery.Find(edus[i], "//div[@data-qa=\"resume-block-education-organization\"]")
			
			if len(years) > 0 {
				ed += htmlquery.InnerText(years[0]) + ඞ
			}
			if len(name) > 0 {
				ed += htmlquery.InnerText(name[0]) + ඞ
			}
			if len(place) > 0 {
				ed += htmlquery.InnerText(place[0])
			}
			
			resume.Education = append(resume.Education, ed)
		}

		// langs 'p[data-qa="resume-block-language-item"]'
		langs := htmlquery.Find(doc, "//p[@data-qa=\"resume-block-language-item\"]")
		for _, n := range langs {
			resume.Langs = htmlquery.InnerText(n)
		}

		// a_edu resume-block-additional-education
		a_edu := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-additional-education\"]//div[@class=\"resume-block-item-gap\"]")
		for _, n := range a_edu {
			asdasdasdasd := ""
			year := htmlquery.Find(n, "//div[@class=\"bloko-column bloko-column_xs-4 bloko-column_s-2 bloko-column_m-2 bloko-column_l-2\"]")
			desc := htmlquery.Find(n, "//div[@data-qa=\"resume-block-education-item\"]")
			if len(year) > 0 {
				asdasdasdasd += htmlquery.InnerText(year[0]) + "ඞ"
			}
			if len(desc) > 0 {
				htmlquery.InnerText(desc[0])
			}

			resume.AdditionalEducation = append(resume.AdditionalEducation, asdasdasdasd)
		}

		// tests_info 'div[data-qa="resume-block-attestation-education"]'
		tests_info := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-attestation-education\"]//div[@class=\"resume-block-item-gap\"]")
		for _, n := range tests_info {
			con := ""
			year := htmlquery.Find(n, "//div[@class=\"bloko-column bloko-column_xs-4 bloko-column_s-2 bloko-column_m-2 bloko-column_l-2\"]")
			desc := htmlquery.Find(n, "//div[@data-qa=\"resume-block-education-item\"]")

			if len(year) > 0 {
				con += htmlquery.InnerText(year[0]) + "ඞ"
			}
			if len(desc) > 0 {
				con += htmlquery.InnerText(desc[0])
			}

			resume.Tests = append(resume.Tests, con)
		}

		// cert = t(soup.find('div', {'class':'resume-certificates'}))
		// ermmmm empty :D
		cert := htmlquery.Find(doc, "//div[@class=\"resume-certificates\"]")
		if len(cert) > 0 {
			resume.Certificates = htmlquery.InnerText(cert[0])
		}

		// add_info = soup.select_one('div[data-qa="resume-block-additional"]')
		add_infos := htmlquery.Find(doc, "//div[@data-qa=\"resume-block-additional\"]//p")
		for _, n := range add_infos {
			resume.AdditionalInfo = htmlquery.InnerText(n)
		}



		if len(resume.Zgr) == 0 {
			resume.Zgr = []string{}
		}


		if len(resume.Experience) == 0 {
			resume.Experience = []string{}
		}

		if len(resume.Skills) == 0 {
			resume.Skills = []string{}
		}


		if len(resume.Education) == 0 {
			resume.Education = []string{}
		}


		if len(resume.AdditionalEducation) == 0 {
			resume.AdditionalEducation = []string{}
		}

		if len(resume.Tests) == 0 {
			resume.Tests = []string{}
		}



		resumes = append(resumes, &resume)
	}

	// check for empty resumes and drop them
	var new_resumes []*pb.Resume
	for _, r := range resumes {
		if r != nil {
			new_resumes = append(new_resumes, r)
		}
	}



	return new_resumes
}