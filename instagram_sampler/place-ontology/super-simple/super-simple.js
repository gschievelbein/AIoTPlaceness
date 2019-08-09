
var simple_chart_config = {
	chart: {
		container: "#OrganiseChart-simple"
	},
	
	nodeStructure: {
		text: { name: "Parent node" },
		children: [
			{
				text: { name: "First child" }
			},
			{
				text: { name: "Second child" }
			}
		]
	}
};

// // // // // // // // // // // // // // // // // // // // // // // // 

/*
var config = {
	container: "#OrganiseChart-simple"
};

var parent_node = {
	text: { name: "Activity" }
};

var first_child = {
	parent: parent_node,
	text: { name: "Indoor" }
};

var second_child = {
	parent: parent_node,
	text: { name: "Outdoor" }
};

var simple_chart_config = [
	config, parent_node,
		first_child, second_child 
];
*/
var simple_chart_config = {
    chart: {
        container: "#OrganiseChart-simple"
    },
    
    nodeStructure: {
        text: { name: "Parent node" },
        children: [
            {
                text: { name: "First child" }
            },
            {
                text: { name: "Second child" }
            }
        ]
    }
};

